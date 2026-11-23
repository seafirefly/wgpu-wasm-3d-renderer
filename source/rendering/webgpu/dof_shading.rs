use crate::engine;
use crate::rendering::common;
use crate::rendering::webgpu::gpu_profiler::GpuPassGroup;
use crate::rendering::webgpu::{
    WebGPUInterface, WebGPUUniqueResources, WEBGPU_CULL_MODE, WEBGPU_FRONT_FACE,
};

// Must match the near / far planes of the perspective projection used by forward and deferred shading.
const PROJECTION_NEAR: f32 = 0.01;
const PROJECTION_FAR: f32 = 100.0;

// depth of field pass --------------------------------------------------------------------------------------

pub fn dof_pass(
    interface: &WebGPUInterface,
    scene: &engine::scene::Scene,
    command_encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    global_resources: &mut WebGPUUniqueResources,
) {
    let _cpu = interface.gpu_profiler.cpu_scope(GpuPassGroup::DepthOfField);
    if global_resources.dof_shading_resource.is_none() {
        global_resources.dof_shading_resource = Some(create_dof_shader_resource(interface));
    }

    update_dof_uniform_buffer(interface, scene, global_resources);

    let resource = global_resources.dof_shading_resource.as_ref().unwrap();
    let mut dof_shading_pass: wgpu::RenderPass<'_> =
        command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Depth of field shading pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: interface.gpu_profiler.timestamp_writes(GpuPassGroup::DepthOfField),
            occlusion_query_set: None,
        });

    dof_shading_pass.set_pipeline(&resource.render_pipeline);
    dof_shading_pass.set_bind_group(0, &resource.texture_bind_group, &[]);
    dof_shading_pass.set_bind_group(1, &resource.uniform_bind_group, &[]);
    dof_shading_pass.draw(0..3, 0..1);
}

// dof shader resource creation and update functions ----------------------------------------------------------------------------

pub struct WebGPUDofShadingResource {
    pub _shader: wgpu::ShaderModule,
    pub uniform_buffer: wgpu::Buffer,
    pub texture_bind_group: wgpu::BindGroup,
    pub uniform_bind_group: wgpu::BindGroup,
    pub render_pipeline: wgpu::RenderPipeline,
}

fn create_dof_shader_resource(interface: &WebGPUInterface) -> WebGPUDofShadingResource {
    let shader: wgpu::ShaderModule =
        interface
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "../../shader/postprocess_dof.wgsl"
                ))),
            });

    let texture_bind_group_layout =
        interface
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Scene (intermediate) texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    // Scene texture sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Depth texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Depth,
                        },
                        count: None,
                    },
                ],
                label: Some("dof_texture_bind_group_layout"),
            });

    let sampler: wgpu::Sampler = interface.device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let texture_bind_group: wgpu::BindGroup =
        interface
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &interface
                                .intermediate_texture
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &interface
                                .depth_texture
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                ],
                label: Some("dof_texture_bind_group"),
            });

    let uniform_buffer = interface.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("DOF Uniform Buffer"),
        size: 4 * std::mem::size_of::<f32>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniform_bind_group_layout =
        interface
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some("dof_uniform_bind_group_layout"),
            });

    let uniform_bind_group: wgpu::BindGroup =
        interface
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                }],
                label: Some("dof_uniform_bind_group"),
            });

    let pipeline_layout: wgpu::PipelineLayout =
        interface
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&texture_bind_group_layout, &uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

    let render_pipeline: wgpu::RenderPipeline =
        interface
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
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
                    targets: &[Some(interface.intermediate_texture_2.format().into())],
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: WEBGPU_FRONT_FACE,
                    cull_mode: Some(WEBGPU_CULL_MODE),
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

    return WebGPUDofShadingResource {
        _shader: shader,
        uniform_buffer,
        texture_bind_group,
        uniform_bind_group,
        render_pipeline,
    };
}

fn update_dof_uniform_buffer(
    interface: &WebGPUInterface,
    scene: &engine::scene::Scene,
    global_resources: &WebGPUUniqueResources,
) {
    let parameters = &scene.parameters;
    interface.queue.write_buffer(
        &global_resources
            .dof_shading_resource
            .as_ref()
            .unwrap()
            .uniform_buffer,
        0,
        bytemuck::cast_slice(&[
            parameters.dof_focus_distance,
            parameters.dof_aperture,
            PROJECTION_NEAR,
            PROJECTION_FAR,
        ]),
    );
}
