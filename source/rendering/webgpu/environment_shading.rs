use crate::engine;
use crate::rendering::common;
use crate::rendering::webgpu::differed_shading::{self, WebGPUDifferedShadingResource};
use crate::rendering::webgpu::gpu_profiler::GpuPassGroup;
use crate::rendering::webgpu::{WebGPUInterface, WebGPUUniqueResources};

// Uniform layout: view_projection (16) + inv_view_projection (16) + camera_position (4) + ssr_params (4) + ambient_light (4)
const ENVIRONMENT_UNIFORM_FLOATS: usize = 44;

// Environment shading pass --------------------------------------------------------------------------

/// The environment pass adds the ambient term and, when IBL is available, the diffuse and specular environment lighting.
/// It is skipped while a debug view replaces the lighting.
pub fn is_environment_pass_active(scene: &engine::scene::Scene) -> bool {
    matches!(
        scene.parameters.scene_shading_type,
        engine::scene::ShadingType::Differed
    ) && scene.parameters.differed_debug_type == 0
}

/// Reads the lit HDR scene (intermediate_texture) and writes it with environment lighting added to intermediate_texture_2.
/// The caller copies the result back to intermediate_texture.
pub fn environment_shading_pass(
    interface: &WebGPUInterface,
    scene: &engine::scene::Scene,
    command_encoder: &mut wgpu::CommandEncoder,
    global_resources: &mut WebGPUUniqueResources,
) {
    let Some(dsr) = global_resources.differed_shading_resource.as_ref() else {
        return;
    };
    if !is_environment_pass_active(scene) {
        return;
    }

    let is_ibl_available = scene.parameters.is_use_environment_pass
        && scene.parameters.is_use_sky_box
        && scene.parameters.is_use_environment_map
        && global_resources.hdr_convertion_resource.is_some();

    let _cpu = interface.gpu_profiler.cpu_scope(GpuPassGroup::EnvironmentPass);
    if global_resources.environment_shading_resource.is_none() {
        global_resources.environment_shading_resource =
            Some(create_environment_shading_resource(interface, dsr));
    }

    let resource = global_resources
        .environment_shading_resource
        .as_ref()
        .unwrap();
    update_environment_shading_uniform(interface, scene, resource, is_ibl_available);

    let mut environment_render_pass =
        command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Environment shading pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &interface
                    .intermediate_texture_2
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: interface
                .gpu_profiler
                .timestamp_writes(GpuPassGroup::EnvironmentPass),
            occlusion_query_set: None,
        });

    environment_render_pass.set_pipeline(&resource.render_pipeline);
    // gbuffers (0), environment uniform (1), IBL (2), lit scene color (3)
    environment_render_pass.set_bind_group(0, &dsr.bind_groups[0], &[]);
    environment_render_pass.set_bind_group(1, &resource.uniform_bind_group, &[]);
    environment_render_pass.set_bind_group(2, &dsr.bind_groups[3], &[]);
    environment_render_pass.set_bind_group(3, &resource.scene_bind_group, &[]);
    environment_render_pass.draw(0..6, 0..1);
}

// Environment shading resource creation and update functions ---------------------------------------

pub struct WebGPUEnvironmentShadingResource {
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub scene_bind_group: wgpu::BindGroup,
    pub render_pipeline: wgpu::RenderPipeline,
}

fn create_environment_shading_resource(
    interface: &WebGPUInterface,
    dsr: &WebGPUDifferedShadingResource,
) -> WebGPUEnvironmentShadingResource {
    let device = &interface.device;

    let shader: wgpu::ShaderModule = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Environment shading shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(concat!(
            include_str!("../../shader/pbr_common.wgsl"),
            "\n",
            include_str!("../../shader/environment_shading.wgsl"),
        ))),
    });

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Environment uniform buffer"),
        size: (ENVIRONMENT_UNIFORM_FLOATS * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniform_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Environment uniform bind group layout"),
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

    let scene_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Environment scene bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

    // The gbuffer and IBL layouts are taken from the deferred pipelines so the bind groups can be shared
    let gbuffer_bind_group_layout = dsr.render_pipeline.get_bind_group_layout(0);
    let ibl_bind_group_layout = &dsr.ibl_bind_group_layout;

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Environment shading pipeline layout"),
        bind_group_layouts: &[
            &gbuffer_bind_group_layout,
            &uniform_bind_group_layout,
            ibl_bind_group_layout,
            &scene_bind_group_layout,
        ],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Environment shading pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some(common::VS_ENTRY_POINT),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(common::FS_ENTRY_POINT),
            compilation_options: Default::default(),
            targets: &[Some(interface.intermediate_texture.format().into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Environment uniform bind group"),
        layout: &uniform_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Environment scene bind group"),
        layout: &scene_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(
                &interface
                    .intermediate_texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
            ),
        }],
    });

    WebGPUEnvironmentShadingResource {
        uniform_buffer,
        uniform_bind_group,
        scene_bind_group,
        render_pipeline,
    }
}

fn update_environment_shading_uniform(
    interface: &WebGPUInterface,
    scene: &engine::scene::Scene,
    resource: &WebGPUEnvironmentShadingResource,
    is_ibl_available: bool,
) {
    // Same view and projection as the deferred gbuffer pass, so the screen projection matches the stored world positions
    let (width, height) = differed_shading::get_canvas_size();
    let aspect_ratio: f32 = width as f32 / height as f32;

    let eye: glam::Vec3 = scene.parameters.eye_location;
    let direction: glam::Vec3 = scene.parameters.eye_direction;
    let view_matrix = glam::Mat4::look_to_rh(eye, direction, glam::Vec3::Z);
    let projection_matrix = glam::Mat4::perspective_rh(
        scene.parameters.fov_y_degrees.to_radians(),
        aspect_ratio,
        0.01,
        100.0,
    );
    let view_projection = projection_matrix * view_matrix;
    let inv_view_projection = view_projection.inverse();

    let ssr_enabled: f32 = if scene.parameters.is_use_ssr {
        1.0
    } else {
        0.0
    };

    let ibl_enabled: f32 = if is_ibl_available { 1.0 } else { 0.0 };
    let ibl_diffuse_enabled: f32 = if scene.parameters.is_use_ibl_diffuse { 1.0 } else { 0.0 };
    let ibl_specular_enabled: f32 = if scene.parameters.is_use_ibl_specular { 1.0 } else { 0.0 };
    let ambient: [f32; 4] = scene.parameters.light_parameters.ambient_light_color;

    let mut uniform_total: Vec<f32> = Vec::new();
    uniform_total.extend_from_slice(&view_projection.to_cols_array());
    uniform_total.extend_from_slice(&inv_view_projection.to_cols_array());
    uniform_total.extend_from_slice(&[eye.x, eye.y, eye.z, 1.0]);
    uniform_total.extend_from_slice(&[ssr_enabled, ibl_enabled, ibl_diffuse_enabled, ibl_specular_enabled]);
    uniform_total.extend_from_slice(&ambient);

    let uniform_ref: &[f32] = uniform_total.as_ref();
    interface.queue.write_buffer(
        &resource.uniform_buffer,
        0,
        bytemuck::cast_slice(uniform_ref),
    );
}
