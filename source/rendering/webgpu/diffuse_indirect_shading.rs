use crate::engine;
use crate::rendering::common;
use crate::rendering::webgpu::deferred_shading::{self, WebGPUDeferredShadingResource};
use crate::rendering::webgpu::gpu_profiler::GpuPassGroup;
use crate::rendering::webgpu::WebGPUInterface;
use std::cell::Cell;

// Uniform layout: view_projection (16) + inv_view_projection (16) + camera_position (4)
// + prev_view_projection (16) + prev_camera_position (4) + params (4) + history (4)
const DIFFUSE_INDIRECT_UNIFORM_FLOATS: usize = 64;
// Maximum sample count of the temporal accumulation. The current frame's weight is 1 / sample count, so this sets the
// slowest blend (1 / max). Larger values are smoother but respond more slowly to changes.
const TEMPORAL_MAX_SAMPLE_COUNT: f32 = 20.0;
// Relative distance tolerance for rejecting disoccluded history and for the depth weight of the denoiser
const DEPTH_TOLERANCE: f32 = 0.05;

// Diffuse indirect pass -----------------------------------------------------------------------------

/// Runs after the BasePass and before the lighting pass. Three passes:
/// 1. trace: screen space bounces from the gbuffers (indirect light, AO, view distance)
/// 2. temporal: accumulates the trace into the previous frame's result (ping-pong)
/// 3. denoise: edge-aware blur of the accumulated result into `resource.indirect_view`, read by the lighting pass
pub fn diffuse_indirect_pass(
    interface: &WebGPUInterface,
    scene: &engine::scene::Scene,
    command_encoder: &mut wgpu::CommandEncoder,
    dsr: &WebGPUDeferredShadingResource,
    resource: &WebGPUDiffuseIndirectResource,
) {
    let _cpu = interface.gpu_profiler.cpu_scope(GpuPassGroup::DiffuseIndirect);
    if !scene.parameters.is_use_diffuse_indirect {
        resource.is_history_valid.set(false);
        clear_indirect_output(interface, command_encoder, resource);
        return;
    }

    let frame = resource.frame_index.get();
    let parity = (frame % 2) as usize;
    update_diffuse_indirect_uniform(interface, scene, resource);
    resource.frame_index.set(frame.wrapping_add(1));

    let gbuffer_bind_group = create_gbuffer_bind_group(interface, dsr, resource);
    let depth_view = interface
        .depth_texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let temporal_bind_group = create_temporal_bind_group(interface, resource, parity, &depth_view);
    let denoise_bind_group = create_denoise_bind_group(interface, dsr, resource, parity);

    {
        let mut trace_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Diffuse indirect trace pass"),
            color_attachments: &[
                Some(clear_attachment(&resource.trace_view, wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 })),
                Some(clear_attachment(&resource.trace_distance_view, wgpu::Color::BLACK)),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: interface
                .gpu_profiler
                .timestamp_writes(GpuPassGroup::DiffuseIndirect),
            occlusion_query_set: None,
        });
        trace_pass.set_pipeline(&resource.trace_pipeline);
        trace_pass.set_bind_group(0, &gbuffer_bind_group, &[]);
        trace_pass.set_bind_group(1, &resource.uniform_bind_group, &[]);
        trace_pass.draw(0..6, 0..1);
    }

    {
        let mut temporal_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Diffuse indirect temporal pass"),
            color_attachments: &[
                Some(clear_attachment(&resource.accumulation_views[parity], wgpu::Color::BLACK)),
                Some(clear_attachment(&resource.accumulation_distance_views[parity], wgpu::Color::BLACK)),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: interface
                .gpu_profiler
                .timestamp_writes(GpuPassGroup::DiffuseIndirect),
            occlusion_query_set: None,
        });
        temporal_pass.set_pipeline(&resource.temporal_pipeline);
        temporal_pass.set_bind_group(0, &temporal_bind_group, &[]);
        temporal_pass.set_bind_group(1, &resource.uniform_bind_group, &[]);
        temporal_pass.draw(0..6, 0..1);
    }

    {
        let mut denoise_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Diffuse indirect denoise pass"),
            color_attachments: &[Some(clear_attachment(&resource.indirect_view, wgpu::Color::BLACK))],
            depth_stencil_attachment: None,
            timestamp_writes: interface
                .gpu_profiler
                .timestamp_writes(GpuPassGroup::DiffuseIndirect),
            occlusion_query_set: None,
        });
        denoise_pass.set_pipeline(&resource.denoise_pipeline);
        denoise_pass.set_bind_group(0, &denoise_bind_group, &[]);
        denoise_pass.set_bind_group(1, &resource.uniform_bind_group, &[]);
        denoise_pass.draw(0..6, 0..1);
    }

    resource.is_history_valid.set(true);
}

/// Copies the lit scene (SceneColor after the lighting pass) into the history texture.
/// The next frame's trace pass uses it as the bounce radiance, so the indirect light lags by one frame.
pub fn save_diffuse_indirect_history(
    interface: &WebGPUInterface,
    command_encoder: &mut wgpu::CommandEncoder,
    resource: &WebGPUDiffuseIndirectResource,
) {
    command_encoder.copy_texture_to_texture(
        interface.intermediate_texture.as_image_copy(),
        resource.history_texture.as_image_copy(),
        interface.intermediate_texture.size(),
    );
}

fn clear_attachment<'a>(
    view: &'a wgpu::TextureView,
    color: wgpu::Color,
) -> wgpu::RenderPassColorAttachment<'a> {
    wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(color),
            store: wgpu::StoreOp::Store,
        },
    }
}

// When disabled, the output is cleared to (rgb 0, ao 1), which leaves the lighting unchanged
fn clear_indirect_output(
    interface: &WebGPUInterface,
    command_encoder: &mut wgpu::CommandEncoder,
    resource: &WebGPUDiffuseIndirectResource,
) {
    let _pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Diffuse indirect clear pass"),
        color_attachments: &[Some(clear_attachment(
            &resource.indirect_view,
            wgpu::Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        ))],
        depth_stencil_attachment: None,
        timestamp_writes: interface
            .gpu_profiler
            .timestamp_writes(GpuPassGroup::DiffuseIndirect),
        occlusion_query_set: None,
    });
}

// Diffuse indirect resource creation and update functions -------------------------------------------

pub struct WebGPUDiffuseIndirectResource {
    trace_view: wgpu::TextureView,
    trace_distance_view: wgpu::TextureView,
    accumulation_views: [wgpu::TextureView; 2],
    accumulation_distance_views: [wgpu::TextureView; 2],
    pub indirect_view: wgpu::TextureView,
    history_texture: wgpu::Texture,
    history_view: wgpu::TextureView,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    trace_pipeline: wgpu::RenderPipeline,
    temporal_bind_group_layout: wgpu::BindGroupLayout,
    temporal_pipeline: wgpu::RenderPipeline,
    denoise_bind_group_layout: wgpu::BindGroupLayout,
    denoise_pipeline: wgpu::RenderPipeline,
    pub lighting_bind_group_layout: wgpu::BindGroupLayout,
    pub lighting_bind_group: wgpu::BindGroup,
    frame_index: Cell<u32>,
    prev_view_projection: Cell<glam::Mat4>,
    prev_camera_position: Cell<glam::Vec3>,
    is_history_valid: Cell<bool>,
}

fn create_pass_texture(
    interface: &WebGPUInterface,
    label: &str,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    interface.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: interface.intermediate_texture.size(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn texture_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn create_fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    targets: &[Option<wgpu::ColorTargetState>],
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(common::VS_ENTRY_POINT),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(common::FS_ENTRY_POINT),
            compilation_options: Default::default(),
            targets,
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

/// Creates the textures, bind group layouts and pipelines. It does not depend on the deferred resources,
/// so it is created before them and the lighting pipeline can reference its bind group layout.
pub fn create_diffuse_indirect_resource(interface: &WebGPUInterface) -> WebGPUDiffuseIndirectResource {
    let device = &interface.device;
    let format = interface.intermediate_texture.format();
    let render_and_sample = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let distance_format = wgpu::TextureFormat::Rg32Float;

    let trace_texture = create_pass_texture(interface, "Diffuse indirect trace texture", format, render_and_sample);
    let trace_distance_texture = create_pass_texture(interface, "Diffuse indirect trace distance texture", distance_format, render_and_sample);
    let accumulation_textures = [0, 1].map(|index| {
        create_pass_texture(interface, &format!("Diffuse indirect accumulation texture {index}"), format, render_and_sample)
    });
    let accumulation_distance_textures = [0, 1].map(|index| {
        create_pass_texture(interface, &format!("Diffuse indirect accumulation distance texture {index}"), distance_format, render_and_sample)
    });
    let indirect_texture = create_pass_texture(interface, "Diffuse indirect texture", format, render_and_sample);
    let history_texture = create_pass_texture(
        interface,
        "Diffuse indirect history texture",
        format,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );

    let view = |texture: &wgpu::Texture| texture.create_view(&wgpu::TextureViewDescriptor::default());
    let history_view = view(&history_texture);

    let trace_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Diffuse indirect trace shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(concat!(
            include_str!("../../shader/pbr_common.wgsl"),
            "\n",
            include_str!("../../shader/diffuse_indirect_common.wgsl"),
            "\n",
            include_str!("../../shader/diffuse_indirect_trace.wgsl"),
        ))),
    });
    let temporal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Diffuse indirect temporal shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(concat!(
            include_str!("../../shader/pbr_common.wgsl"),
            "\n",
            include_str!("../../shader/diffuse_indirect_common.wgsl"),
            "\n",
            include_str!("../../shader/diffuse_indirect_temporal.wgsl"),
        ))),
    });
    let denoise_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Diffuse indirect denoise shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(concat!(
            include_str!("../../shader/pbr_common.wgsl"),
            "\n",
            include_str!("../../shader/diffuse_indirect_common.wgsl"),
            "\n",
            include_str!("../../shader/diffuse_indirect_denoise.wgsl"),
        ))),
    });

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Diffuse indirect uniform buffer"),
        size: (DIFFUSE_INDIRECT_UNIFORM_FLOATS * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Diffuse indirect uniform bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Diffuse indirect uniform bind group"),
        layout: &uniform_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    // trace: gbuffers (normal, depth, albedo, metallic) and history
    let gbuffer_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Diffuse indirect gbuffer bind group layout"),
        entries: &[
            texture_entry(0, true),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            texture_entry(2, true),
            texture_entry(3, true),
            texture_entry(4, true),
        ],
    });
    // temporal: raw trace, previous accumulation (both with view distance), depth
    let temporal_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Diffuse indirect temporal bind group layout"),
        entries: &[
            texture_entry(0, true),
            texture_entry(1, false),
            texture_entry(2, true),
            texture_entry(3, false),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    // denoise: accumulated result, view distance, normal
    let denoise_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Diffuse indirect denoise bind group layout"),
        entries: &[texture_entry(0, true), texture_entry(1, false), texture_entry(2, true)],
    });

    let make_layout = |label: &str, group0: &wgpu::BindGroupLayout| {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[group0, &uniform_bind_group_layout],
            push_constant_ranges: &[],
        })
    };
    let trace_layout = make_layout("Diffuse indirect trace pipeline layout", &gbuffer_bind_group_layout);
    let temporal_layout = make_layout("Diffuse indirect temporal pipeline layout", &temporal_bind_group_layout);
    let denoise_layout = make_layout("Diffuse indirect denoise pipeline layout", &denoise_bind_group_layout);

    let trace_pipeline = create_fullscreen_pipeline(
        device,
        "Diffuse indirect trace pipeline",
        &trace_layout,
        &trace_shader,
        &[Some(format.into()), Some(distance_format.into())],
    );
    let temporal_pipeline = create_fullscreen_pipeline(
        device,
        "Diffuse indirect temporal pipeline",
        &temporal_layout,
        &temporal_shader,
        &[Some(format.into()), Some(distance_format.into())],
    );
    let denoise_pipeline = create_fullscreen_pipeline(
        device,
        "Diffuse indirect denoise pipeline",
        &denoise_layout,
        &denoise_shader,
        &[Some(format.into())],
    );

    let indirect_view = view(&indirect_texture);
    let lighting_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Diffuse indirect lighting bind group layout"),
        entries: &[texture_entry(0, true)],
    });
    let lighting_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Diffuse indirect lighting bind group"),
        layout: &lighting_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&indirect_view),
        }],
    });

    let [accumulation_0, accumulation_1] = accumulation_textures;
    let [accumulation_distance_0, accumulation_distance_1] = accumulation_distance_textures;

    WebGPUDiffuseIndirectResource {
        trace_view: view(&trace_texture),
        trace_distance_view: view(&trace_distance_texture),
        accumulation_views: [view(&accumulation_0), view(&accumulation_1)],
        accumulation_distance_views: [view(&accumulation_distance_0), view(&accumulation_distance_1)],
        indirect_view,
        history_texture,
        history_view,
        uniform_buffer,
        uniform_bind_group,
        gbuffer_bind_group_layout,
        trace_pipeline,
        temporal_bind_group_layout,
        temporal_pipeline,
        denoise_bind_group_layout,
        denoise_pipeline,
        lighting_bind_group_layout,
        lighting_bind_group,
        frame_index: Cell::new(0),
        prev_view_projection: Cell::new(glam::Mat4::IDENTITY),
        prev_camera_position: Cell::new(glam::Vec3::ZERO),
        is_history_valid: Cell::new(false),
    }
}

fn create_gbuffer_bind_group(
    interface: &WebGPUInterface,
    dsr: &WebGPUDeferredShadingResource,
    resource: &WebGPUDiffuseIndirectResource,
) -> wgpu::BindGroup {
    let depth_view = interface
        .depth_texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    interface
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Diffuse indirect gbuffer bind group"),
            layout: &resource.gbuffer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dsr.gbuffer_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&dsr.gbuffer_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&dsr.gbuffer_views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&resource.history_view),
                },
            ],
        })
}

fn create_temporal_bind_group(
    interface: &WebGPUInterface,
    resource: &WebGPUDiffuseIndirectResource,
    parity: usize,
    depth_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    let previous = 1 - parity;
    interface
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Diffuse indirect temporal bind group"),
            layout: &resource.temporal_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&resource.trace_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&resource.trace_distance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&resource.accumulation_views[previous]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&resource.accumulation_distance_views[previous]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
            ],
        })
}

fn create_denoise_bind_group(
    interface: &WebGPUInterface,
    dsr: &WebGPUDeferredShadingResource,
    resource: &WebGPUDiffuseIndirectResource,
    parity: usize,
) -> wgpu::BindGroup {
    interface
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Diffuse indirect denoise bind group"),
            layout: &resource.denoise_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&resource.accumulation_views[parity]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&resource.accumulation_distance_views[parity]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&dsr.gbuffer_views[0]),
                },
            ],
        })
}

fn update_diffuse_indirect_uniform(
    interface: &WebGPUInterface,
    scene: &engine::scene::Scene,
    resource: &WebGPUDiffuseIndirectResource,
) {
    // Same view and projection as the deferred gbuffer pass, so the screen projection matches the stored world positions
    let (width, height) = deferred_shading::get_canvas_size();
    let aspect_ratio: f32 = width as f32 / height as f32;

    let eye: glam::Vec3 = scene.parameters.eye_location;
    let view_matrix = glam::Mat4::look_to_rh(eye, scene.parameters.eye_direction, glam::Vec3::Z);
    let projection_matrix = glam::Mat4::perspective_rh(
        scene.parameters.fov_y_degrees.to_radians(),
        aspect_ratio,
        0.01,
        100.0,
    );
    let view_projection = projection_matrix * view_matrix;
    let inv_view_projection = view_projection.inverse();

    let prev_view_projection = resource.prev_view_projection.get();
    let prev_camera_position = resource.prev_camera_position.get();
    let history_valid: f32 = if resource.is_history_valid.get() { 1.0 } else { 0.0 };

    let mut uniform_total: Vec<f32> = Vec::new();
    uniform_total.extend_from_slice(&view_projection.to_cols_array());
    uniform_total.extend_from_slice(&inv_view_projection.to_cols_array());
    uniform_total.extend_from_slice(&[eye.x, eye.y, eye.z, 1.0]);
    uniform_total.extend_from_slice(&prev_view_projection.to_cols_array());
    uniform_total.extend_from_slice(&[prev_camera_position.x, prev_camera_position.y, prev_camera_position.z, 1.0]);
    uniform_total.extend_from_slice(&[
        scene.parameters.diffuse_indirect_max_distance,
        scene.parameters.ambient_occlusion_strength,
        scene.parameters.diffuse_indirect_intensity,
        resource.frame_index.get() as f32,
    ]);
    uniform_total.extend_from_slice(&[history_valid, TEMPORAL_MAX_SAMPLE_COUNT, DEPTH_TOLERANCE, 0.0]);

    resource.prev_view_projection.set(view_projection);
    resource.prev_camera_position.set(eye);

    interface.queue.write_buffer(
        &resource.uniform_buffer,
        0,
        bytemuck::cast_slice(uniform_total.as_slice()),
    );
}
