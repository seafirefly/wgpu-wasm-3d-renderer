use crate::engine::scene::{
    CompositeViewTransform, Scene, SceneModel, ShadingType, SkyBoxResolution, SkyBoxType,
    ToneMappingType,
};
use crate::rendering::webgpu::gpu_profiler::{CpuTimings, GpuPassGroup, GpuTimings};
use crate::types::Shared;
use super::ui_elements::{
    accordion, accordion_value, append, checkbox_row, color_picker, dialog, div, element,
    on_select, select, slider_row, stat_row, value_wrapper, widget_row, Slider,
};
use wasm_bindgen::JsCast;

const DIRECTIONAL_AXES: [(&str, &str); 3] = [
    ("directional-range-x", "X"),
    ("directional-range-y", "Y"),
    ("directional-range-z", "Z"),
];

const CAMERA_POSITION_AXES: [(&str, &str); 3] = [
    ("camera-position-x", "X"),
    ("camera-position-y", "Y"),
    ("camera-position-z", "Z"),
];

const CAMERA_DIRECTION_AXES: [(&str, &str); 3] = [
    ("camera-direction-x", "X"),
    ("camera-direction-y", "Y"),
    ("camera-direction-z", "Z"),
];

const AMBIENT_CHANNELS: [(&str, &str, &str, &str, usize); 4] = [
    ("ambient-range-r", "R", "0.0", "1.0", 0),
    ("ambient-range-g", "G", "0.0", "1.0", 1),
    ("ambient-range-b", "B", "0.0", "1.0", 2),
    ("ambient-range-intensity", "Intensity", "0.0", "10.0", 3),
];

#[cfg(debug_assertions)]
const SCENE_MODELS: [(&str, SceneModel); 3] = [
    ("cornell_box", SceneModel::CornellBox),
    ("torus", SceneModel::Torus),
    ("gltf_logo", SceneModel::GltfLogo),
];

#[cfg(not(debug_assertions))]
const SCENE_MODELS: [(&str, SceneModel); 1] =
    [("cornell_box", SceneModel::CornellBox)];

const SKY_BOX_OPTIONS: [(&str, Option<SkyBoxType>); 4] = [
    ("none", None),
    ("kloofendal", Some(SkyBoxType::Kloofendal48dPartlyCloudy)),
    ("studio", Some(SkyBoxType::StudioSmall09)),
    ("fireplace", Some(SkyBoxType::Fireplace)),
];

#[cfg(debug_assertions)]
const SKY_BOX_RESOLUTIONS: [(&str, SkyBoxResolution); 3] = [
    ("1k", SkyBoxResolution::R1k),
    ("2k", SkyBoxResolution::R2k),
    ("4k", SkyBoxResolution::R4k),
];

#[cfg(not(debug_assertions))]
const SKY_BOX_RESOLUTIONS: [(&str, SkyBoxResolution); 1] = [
    ("1k", SkyBoxResolution::R1k),
];

const RENDER_TYPES: [(&str, ShadingType); 2] = [
    ("deferred", ShadingType::Deferred),
    ("forward*", ShadingType::Forward),
];

const FORWARD_DISPLAY: [(&str, u8); 2] = [("render", 0), ("normal", 1)];

const BUFFER_TYPES: [(&str, u8); 7] = [
    ("render", 0),
    ("normal", 1),
    ("depth", 2),
    ("albedo", 3),
    ("metallic", 4),
    ("roughness", 5),
    ("specular", 6),
];

const IBL_DEBUG_TYPES: [(&str, u8); 7] = [
    ("off", 0),
    ("irradiance", 1),
    ("prefilter 0", 2),
    ("prefilter 1", 3),
    ("prefilter 2", 4),
    ("prefilter 3", 5),
    ("prefilter 4", 6),
];

const TONE_MAPPINGS: [(&str, ToneMappingType); 4] = [
    ("off", ToneMappingType::Off),
    ("aces", ToneMappingType::Aces),
    ("filmic", ToneMappingType::Filmic),
    ("agx", ToneMappingType::Agx),
];

const VIEW_TRANSFORMS: [(&str, CompositeViewTransform); 3] = [
    ("standard", CompositeViewTransform::Standard),
    ("agx", CompositeViewTransform::Agx),
    ("filmic", CompositeViewTransform::Filmic),
];

pub fn create_control_panel(scene: &Shared<Scene>) {
    let wrapper = element("div");
    wrapper.set_id("dialog-wrapper");

    environment(&wrapper, scene);
    lighting_pass(&wrapper, scene);
    environment_pass(&wrapper, scene);
    postprocess(&wrapper, scene);
    overlay(&wrapper, scene);
    performance(&wrapper);

    append(&gloo::utils::body(), &wrapper);
    set_skybox_dependents_disabled(!scene.borrow().parameters.is_use_sky_box);
}

pub fn update_performance_statistics(average_frame_time_ms: f64, frames_per_second: f64) {
    update_statistic(
        "frame-time-analytics-value",
        &format!("{average_frame_time_ms:.2} ms"),
    );
    update_statistic(
        "fps-analytics-value",
        &format!("{frames_per_second:.1} FPS"),
    );
}

pub fn update_scene_statistics(objects: usize, materials: usize, lights: usize) {
    update_statistic("objects-analytics-value", &objects.to_string());
    update_statistic("materials-analytics-value", &materials.to_string());
    update_statistic("lights-analytics-value", &lights.to_string());
}

/// Reflects the camera pose changed by user input (mouse/keyboard) into the camera sliders.
pub fn update_camera_controls(eye: glam::Vec3, direction: glam::Vec3) {
    for (index, &(id, _)) in CAMERA_POSITION_AXES.iter().enumerate() {
        update_slider(id, eye[index]);
    }
    for (index, &(id, _)) in CAMERA_DIRECTION_AXES.iter().enumerate() {
        update_slider(id, direction[index]);
    }
}

fn update_slider(id: &str, value: f32) {
    let input = gloo::utils::document()
        .get_element_by_id(id)
        .expect("Camera slider element should exist")
        .unchecked_into::<web_sys::HtmlInputElement>();
    input.set_value(&value.to_string());
    if let Some(text) = gloo::utils::document().get_element_by_id(&format!("{id}-text")) {
        text.set_text_content(Some(&input.value()));
    }
}

fn update_statistic(id: &str, value: &str) {
    gloo::utils::document()
        .get_element_by_id(id)
        .expect("Performance statistic element should exist")
        .set_text_content(Some(value));
}

/// Updates GPU pass timings. `timings` is `None` until the first readback completes.
/// CPU encoding timings are shown regardless of GPU timestamp support.
pub fn update_gpu_statistics(
    supported: bool,
    timings: Option<GpuTimings>,
    cpu_timings: Option<CpuTimings>,
) {
    if !supported {
        for index in 0..GpuPassGroup::ALL.len() {
            update_statistic(&gpu_pass_row_id(index), "N/A");
        }
        update_statistic(GPU_TOTAL_ROW_ID, "N/A");
    } else {
        for index in 0..GpuPassGroup::ALL.len() {
            let text = match timings.and_then(|t| t.group_ms[index]) {
                Some(ms) => format!("{ms:.2} ms"),
                None => "-".to_string(),
            };
            update_statistic(&gpu_pass_row_id(index), &text);
        }
        let total = match timings {
            Some(t) => format!("{:.2} ms", t.total_ms),
            None => "-".to_string(),
        };
        update_statistic(GPU_TOTAL_ROW_ID, &total);
    }

    for index in 0..GpuPassGroup::ALL.len() {
        let text = match cpu_timings.and_then(|t| t.group_ms[index]) {
            Some(ms) => format!("{ms:.2} ms"),
            None => "-".to_string(),
        };
        update_statistic(&cpu_pass_row_id(index), &text);
    }
    let cpu_total = match cpu_timings {
        Some(t) => format!("{:.2} ms", t.total_ms),
        None => "-".to_string(),
    };
    update_statistic(CPU_TOTAL_ROW_ID, &cpu_total);

    let gpu_label = match timings {
        Some(t) => format!("GPU ({:.2} ms)", t.total_ms),
        None => "GPU".to_string(),
    };
    update_statistic("accordion-performance-gpu-label", &gpu_label);

    let cpu_label = match cpu_timings {
        Some(t) => format!("CPU ({:.2} ms)", t.total_ms),
        None => "CPU".to_string(),
    };
    update_statistic("accordion-performance-cpu-label", &cpu_label);

    let gpu_lighting = ms_text(timings.and_then(|t| group_sum_ms(&t.group_ms, is_lighting)));
    update_statistic("accordion-performance-gpu-lighting-value", &gpu_lighting);

    let cpu_lighting = ms_text(cpu_timings.and_then(|t| group_sum_ms(&t.group_ms, is_lighting)));
    update_statistic("accordion-performance-cpu-lighting-value", &cpu_lighting);

    let gpu_postprocess = ms_text(timings.and_then(|t| group_sum_ms(&t.group_ms, is_postprocess)));
    update_statistic("accordion-performance-gpu-postprocess-value", &gpu_postprocess);

    let cpu_postprocess =
        ms_text(cpu_timings.and_then(|t| group_sum_ms(&t.group_ms, is_postprocess)));
    update_statistic("accordion-performance-cpu-postprocess-value", &cpu_postprocess);

    let gpu_overlay = ms_text(timings.and_then(|t| group_sum_ms(&t.group_ms, is_overlay)));
    update_statistic("accordion-performance-gpu-overlay-value", &gpu_overlay);

    let cpu_overlay = ms_text(cpu_timings.and_then(|t| group_sum_ms(&t.group_ms, is_overlay)));
    update_statistic("accordion-performance-cpu-overlay-value", &cpu_overlay);
}

const GPU_TOTAL_ROW_ID: &str = "gpu-total-analytics-value";

fn gpu_pass_row_id(index: usize) -> String {
    format!("gpu-pass-{index}-analytics-value")
}

const CPU_TOTAL_ROW_ID: &str = "cpu-total-analytics-value";

fn cpu_pass_row_id(index: usize) -> String {
    format!("cpu-pass-{index}-analytics-value")
}

fn environment(parent: &web_sys::Element, scene: &Shared<Scene>) {
    let content = dialog(parent, "world", "World");

    let model = select(
        "scene-model-select",
        &SCENE_MODELS,
        Some(scene.borrow().parameters.scene_model),
    );
    on_select(scene, &model, &SCENE_MODELS, |s, v| {
        s.parameters.requested_scene_model = Some(v);
    });
    append(&content, &widget_row("Scene", &[&value_wrapper(&model)]));
    skybox(&content, scene);

    let directional = accordion(&content, "directional", "Directional Light", true);
    for (index, &(id, label)) in DIRECTIONAL_AXES.iter().enumerate() {
        let value = scene
            .borrow()
            .parameters
            .light_parameters
            .directional_light_angle[index];
        let slider = Slider {
            id,
            label,
            min: "-1.0",
            max: "1.0",
            value,
        };
        append(
            &directional,
            &slider_row(scene, slider, move |s, v| {
                s.parameters.light_parameters.directional_light_angle[index] = v;
            }),
        );
    }
    let value = scene
        .borrow()
        .parameters
        .light_parameters
        .directional_light_intensity;
    let slider = Slider {
        id: "directional-range-intensity",
        label: "Intensity",
        min: "0.0",
        max: "10.0",
        value,
    };
    append(
        &directional,
        &slider_row(scene, slider, |s, v| {
            s.parameters.light_parameters.directional_light_intensity = v;
        }),
    );

    let ambient = accordion(&content, "ambient", "Ambient Light", true);
    for &(id, label, min, max, index) in &AMBIENT_CHANNELS {
        let value = scene
            .borrow()
            .parameters
            .light_parameters
            .ambient_light_color[index];
        let slider = Slider {
            id,
            label,
            min,
            max,
            value,
        };
        append(
            &ambient,
            &slider_row(scene, slider, move |s, v| {
                s.parameters.light_parameters.ambient_light_color[index] = v;
            }),
        );
    }

    camera(&content, scene);

    let information = accordion(&content, "information", "Information", true);
    let (objects, materials, lights) = {
        let scene = scene.borrow();
        (
            scene.objects.len(),
            scene.materials.len(),
            scene.emissive_lights.len(),
        )
    };
    append(
        &information,
        &stat_row("Objects", "objects-analytics-value", objects),
    );
    append(
        &information,
        &stat_row("Materials", "materials-analytics-value", materials),
    );
    append(
        &information,
        &stat_row("Lights", "lights-analytics-value", lights),
    );
}

fn camera(parent: &web_sys::Element, scene: &Shared<Scene>) {
    let content = accordion(parent, "camera", "Camera", true);

    let position = accordion(&content, "camera-position", "Position", true);
    for (index, &(id, label)) in CAMERA_POSITION_AXES.iter().enumerate() {
        let value = scene.borrow().parameters.eye_location[index];
        let slider = Slider {
            id,
            label,
            min: "-20.0",
            max: "20.0",
            value,
        };
        append(
            &position,
            &slider_row(scene, slider, move |s, v| {
                s.parameters.eye_location[index] = v;
            }),
        );
    }

    let direction = accordion(&content, "camera-direction", "Direction", true);
    for (index, &(id, label)) in CAMERA_DIRECTION_AXES.iter().enumerate() {
        let value = scene.borrow().parameters.eye_direction[index];
        let slider = Slider {
            id,
            label,
            min: "-3.0",
            max: "3.0",
            value,
        };
        append(
            &direction,
            &slider_row(scene, slider, move |s, v| {
                s.parameters.eye_direction[index] = v;
            }),
        );
    }

    let value = scene.borrow().parameters.fov_y_degrees;
    let slider = Slider {
        id: "camera-fov-range",
        label: "FOV",
        min: "10.0",
        max: "120.0",
        value,
    };
    append(
        &content,
        &slider_row(scene, slider, |s, v| s.parameters.fov_y_degrees = v),
    );
}

fn lighting_pass(parent: &web_sys::Element, scene: &Shared<Scene>) {
    let content = dialog(parent, "lightingpass", "Lighting Pass");
    let shading = scene.borrow().parameters.scene_shading_type;

    let forward_wrapper = wrapper_div("forward-wrapper", shading == ShadingType::Forward);
    let deferred_wrapper = wrapper_div("deferred-wrapper", shading == ShadingType::Deferred);

    let render_type = select("render-type-select", &RENDER_TYPES, Some(shading));
    {
        let forward_wrapper = forward_wrapper.clone();
        let deferred_wrapper = deferred_wrapper.clone();
        on_select(scene, &render_type, &RENDER_TYPES, move |s, shading| {
            s.parameters.scene_shading_type = shading;
            let is_forward = shading == ShadingType::Forward;
            forward_wrapper.set_class_name(wrapper_class(is_forward));
            deferred_wrapper.set_class_name(wrapper_class(!is_forward));
        });
    }
    append(&content, &widget_row("Rendering type", &[&render_type]));
    let group = div("widget-wrapper");
    append(&content, &group);

    let forward_display = select("forward-display-select", &FORWARD_DISPLAY, None);
    on_select(scene, &forward_display, &FORWARD_DISPLAY, |s, v| {
        s.parameters.forward_debug_type = v;
    });
    append(&forward_wrapper, &widget_row("Out", &[&forward_display]));
    append(&group, &forward_wrapper);

    let buffer_type = select("buffer-type-select", &BUFFER_TYPES, None);
    on_select(scene, &buffer_type, &BUFFER_TYPES, |s, v| {
        s.parameters.deferred_debug_type = v;
    });
    append(&deferred_wrapper, &widget_row("Out", &[&buffer_type]));
    append(&group, &deferred_wrapper);

    let bg = scene.borrow().parameters.background_color;
    let hex = format!(
        "#{:02X}{:02X}{:02X}",
        (bg[0] * 255.0) as u32,
        (bg[1] * 255.0) as u32,
        (bg[2] * 255.0) as u32,
    );
    let clear_color = color_picker(scene, "background-color-picker", &hex, |s, value| {
        let rgb = u32::from_str_radix(value.trim_start_matches('#'), 16)
            .unwrap()
            .to_be_bytes();
        s.parameters.background_color = [
            rgb[1] as f32 / 255.0,
            rgb[2] as f32 / 255.0,
            rgb[3] as f32 / 255.0,
            1.0,
        ];
    });
    append(&group, &widget_row("Clear Color", &[&clear_color]));

    let diffuse_indirect = accordion(&content, "diffuse-indirect", "Diffuse Indirect (SSGI / AO)", true);
    let checked = scene.borrow().parameters.is_use_diffuse_indirect;
    append(
        &diffuse_indirect,
        &checkbox_row(scene, "Active", "diffuse-indirect-active", checked, |s, v| {
            s.parameters.is_use_diffuse_indirect = v;
        }),
    );
    let value = scene.borrow().parameters.diffuse_indirect_max_distance;
    let slider = Slider {
        id: "diffuse-indirect-distance-range",
        label: "Distance",
        min: "0.5",
        max: "20.0",
        value,
    };
    append(
        &diffuse_indirect,
        &slider_row(scene, slider, |s, v| {
            s.parameters.diffuse_indirect_max_distance = v;
        }),
    );
    let value = scene.borrow().parameters.diffuse_indirect_intensity;
    let slider = Slider {
        id: "diffuse-indirect-intensity-range",
        label: "Intensity",
        min: "0.0",
        max: "4.0",
        value,
    };
    append(
        &diffuse_indirect,
        &slider_row(scene, slider, |s, v| {
            s.parameters.diffuse_indirect_intensity = v;
        }),
    );
    let value = scene.borrow().parameters.ambient_occlusion_strength;
    let slider = Slider {
        id: "ambient-occlusion-strength-range",
        label: "AO",
        min: "0.0",
        max: "1.0",
        value,
    };
    append(
        &diffuse_indirect,
        &slider_row(scene, slider, |s, v| {
            s.parameters.ambient_occlusion_strength = v;
        }),
    );
}

fn wrapper_class(active: bool) -> &'static str {
    if active {
        ""
    } else {
        "widget-wrapper-hidden"
    }
}

fn wrapper_div(id: &str, active: bool) -> web_sys::Element {
    let wrapper = div(if active { "" } else { "widget-wrapper-hidden" });
    wrapper.set_id(id);
    wrapper
}

fn skybox(content: &web_sys::Element, scene: &Shared<Scene>) {
    let (is_use_sky_box, sky_box_type) = {
        let scene = scene.borrow();
        (scene.parameters.is_use_sky_box, scene.parameters.sky_box_type)
    };
    let sky_box = select(
        "sky-box-select",
        &SKY_BOX_OPTIONS,
        Some(is_use_sky_box.then_some(sky_box_type)),
    );
    on_select(scene, &sky_box, &SKY_BOX_OPTIONS, |s, v| {
        s.parameters.is_use_sky_box = v.is_some();
        if let Some(sky_box_type) = v {
            s.parameters.requested_sky_box_type = Some(sky_box_type);
        }
        set_skybox_dependents_disabled(v.is_none());
    });
    append(content, &widget_row("Skybox", &[&value_wrapper(&sky_box)]));

    let resolution_wrapper = div("widget-wrapper");
    let sky_box_resolution = select(
        "sky-box-resolution-select",
        &SKY_BOX_RESOLUTIONS,
        Some(scene.borrow().parameters.sky_box_resolution),
    );
    on_select(scene, &sky_box_resolution, &SKY_BOX_RESOLUTIONS, |s, v| {
        s.parameters.requested_sky_box_resolution = Some(v);
    });
    append(
        &resolution_wrapper,
        &widget_row("Resolution", &[&value_wrapper(&sky_box_resolution)]),
    );
    append(content, &resolution_wrapper);
}


fn set_skybox_dependents_disabled(disabled: bool) {
    for id in [
        "ibl-diffuse-active",
        "ibl-specular-active",
        "ibl-debug-select",
        "sky-box-resolution-select",
    ] {
        let element = gloo::utils::document()
            .get_element_by_id(id)
            .expect("IBL control element should exist");
        if disabled {
            element.set_attribute("disabled", "").unwrap();
        } else {
            element.remove_attribute("disabled").unwrap();
        }
    }
}

fn environment_pass(parent: &web_sys::Element, scene: &Shared<Scene>) {
    let content = dialog(parent, "environment-pass", "Environment Pass");
    let checked = scene.borrow().parameters.is_use_environment_pass;
    append(
        &content,
        &checkbox_row(scene, "Active", "environment-pass-active", checked, |s, v| {
            s.parameters.is_use_environment_pass = v;
        }),
    );

    let checked = scene.borrow().parameters.is_use_ibl_diffuse;
    append(
        &content,
        &checkbox_row(scene, "IBL diffuse", "ibl-diffuse-active", checked, |s, v| {
            s.parameters.is_use_ibl_diffuse = v;
        }),
    );

    let checked = scene.borrow().parameters.is_use_ibl_specular;
    append(
        &content,
        &checkbox_row(scene, "IBL specular", "ibl-specular-active", checked, |s, v| {
            s.parameters.is_use_ibl_specular = v;
        }),
    );

    let ibl_debug = select(
        "ibl-debug-select",
        &IBL_DEBUG_TYPES,
        Some(scene.borrow().parameters.ibl_debug_type),
    );
    on_select(scene, &ibl_debug, &IBL_DEBUG_TYPES, |s, v| {
        s.parameters.ibl_debug_type = v;
    });
    append(&content, &widget_row("IBL debug", &[&value_wrapper(&ibl_debug)]));

    let checked = scene.borrow().parameters.is_use_ssr;
    append(
        &content,
        &checkbox_row(scene, "SSR", "ssr-active", checked, |s, v| {
            s.parameters.is_use_ssr = v;
        }),
    );
}

fn postprocess(parent: &web_sys::Element, scene: &Shared<Scene>) {
    let content = dialog(parent, "postprocess", "Postprocess");
    dof(&content, scene);
    bloom(&content, scene);
    composite(&content, scene);
}

fn dof(parent: &web_sys::Element, scene: &Shared<Scene>) {
    let content = accordion(parent, "dof", "Depth of Field", true);

    let checked = scene.borrow().parameters.is_use_dof;
    append(
        &content,
        &checkbox_row(scene, "Active", "dof-active", checked, |s, v| {
            s.parameters.is_use_dof = v;
        }),
    );

    let value = scene.borrow().parameters.dof_focus_distance;
    let slider = Slider {
        id: "dof-focus-range",
        label: "Focus",
        min: "0.5",
        max: "20.0",
        value,
    };
    append(
        &content,
        &slider_row(scene, slider, |s, v| s.parameters.dof_focus_distance = v),
    );

    let value = scene.borrow().parameters.dof_aperture;
    let slider = Slider {
        id: "dof-aperture-range",
        label: "Aperture",
        min: "0.0",
        max: "0.1",
        value,
    };
    append(
        &content,
        &slider_row(scene, slider, |s, v| s.parameters.dof_aperture = v),
    );
}

fn bloom(parent: &web_sys::Element, scene: &Shared<Scene>) {
    let content = accordion(parent, "bloom", "Bloom", true);

    let checked = scene.borrow().parameters.is_use_bloom;
    append(
        &content,
        &checkbox_row(scene, "Active", "bloom-active", checked, |s, v| {
            s.parameters.is_use_bloom = v;
        }),
    );

    let value = scene.borrow().parameters.bloom_threshold;
    let slider = Slider {
        id: "threshold-range",
        label: "Threshold",
        min: "0.0",
        max: "2.0",
        value,
    };
    append(
        &content,
        &slider_row(scene, slider, |s, v| s.parameters.bloom_threshold = v),
    );
}

fn composite(parent: &web_sys::Element, scene: &Shared<Scene>) {
    let content = accordion(parent, "composite", "Composite", true);

    let checked = scene.borrow().parameters.is_use_composite;
    append(
        &content,
        &checkbox_row(scene, "Active", "composite-active", checked, |s, v| {
            s.parameters.is_use_composite = v;
        }),
    );

    let value = scene.borrow().parameters.composite_exposure;
    let slider = Slider {
        id: "composite-exposure-range",
        label: "Exposure",
        min: "-3.0",
        max: "3.0",
        value,
    };
    append(
        &content,
        &slider_row(scene, slider, |s, v| s.parameters.composite_exposure = v),
    );

    let value = scene.borrow().parameters.composite_saturation;
    let slider = Slider {
        id: "composite-saturation-range",
        label: "Saturation",
        min: "0.0",
        max: "2.0",
        value,
    };
    append(
        &content,
        &slider_row(scene, slider, |s, v| s.parameters.composite_saturation = v),
    );

    let tone_mapping = select(
        "composite-tone-mapping-select",
        &TONE_MAPPINGS,
        Some(scene.borrow().parameters.tone_mapping_type),
    );
    on_select(scene, &tone_mapping, &TONE_MAPPINGS, |s, v| {
        s.parameters.tone_mapping_type = v;
    });
    append(
        &content,
        &widget_row("Tone Mapping", &[&value_wrapper(&tone_mapping)]),
    );

    let value = scene.borrow().parameters.composite_highlight_rolloff;
    let slider = Slider {
        id: "composite-highlight-rolloff-range",
        label: "Soft",
        min: "0.0",
        max: "1.0",
        value,
    };
    append(
        &content,
        &slider_row(scene, slider, |s, v| {
            s.parameters.composite_highlight_rolloff = v
        }),
    );

    let value = scene.borrow().parameters.composite_white_point;
    let slider = Slider {
        id: "composite-white-point-range",
        label: "White",
        min: "0.1",
        max: "4.0",
        value,
    };
    append(
        &content,
        &slider_row(scene, slider, |s, v| s.parameters.composite_white_point = v),
    );

    let view_transform = select(
        "composite-view-transform-select",
        &VIEW_TRANSFORMS,
        Some(scene.borrow().parameters.composite_view_transform),
    );
    on_select(scene, &view_transform, &VIEW_TRANSFORMS, |s, v| {
        s.parameters.composite_view_transform = v;
    });
    append(
        &content,
        &widget_row("View Trans", &[&value_wrapper(&view_transform)]),
    );

    let checked = scene.borrow().parameters.composite_use_dither;
    append(
        &content,
        &checkbox_row(
            scene,
            "Dithered",
            "composite-dither-active",
            checked,
            |s, v| {
                s.parameters.composite_use_dither = v;
            },
        ),
    );

    let checked = scene.borrow().parameters.is_use_gamma_correction;
    append(
        &content,
        &checkbox_row(
            scene,
            "Gamma",
            "gamma-correction-active",
            checked,
            |s, v| {
                s.parameters.is_use_gamma_correction = v;
            },
        ),
    );
}

fn overlay(parent: &web_sys::Element, scene: &Shared<Scene>) {
    let content = dialog(parent, "overlay", "Overlay");
    let grid = accordion(&content, "grid", "Grid", true);

    let checked = scene.borrow().parameters.is_use_grid;
    append(
        &grid,
        &checkbox_row(scene, "Active", "grid-active", checked, |s, v| {
            s.parameters.is_use_grid = v;
        }),
    );
}

fn performance(parent: &web_sys::Element) {
    let content = dialog(parent, "performance", "Performance");

    append(
        &content,
        &stat_row("Frame time", "frame-time-analytics-value", 0),
    );
    append(&content, &stat_row("FPS", "fps-analytics-value", 0));
    let cpu = accordion(&content, "performance-cpu", "CPU", true);
    pass_rows(&cpu, "cpu", cpu_pass_row_id);
    append(&cpu, &stat_row("Total", CPU_TOTAL_ROW_ID, 0));

    let gpu = accordion(&content, "performance-gpu", "GPU", true);
    pass_rows(&gpu, "gpu", gpu_pass_row_id);
    append(&gpu, &stat_row("Total", GPU_TOTAL_ROW_ID, 0));
}

fn is_postprocess(group: GpuPassGroup) -> bool {
    matches!(
        group,
        GpuPassGroup::DepthOfField | GpuPassGroup::Bloom | GpuPassGroup::Composite
    )
}

fn is_overlay(group: GpuPassGroup) -> bool {
    matches!(group, GpuPassGroup::LineGrid)
}

fn group_sum_ms(group_ms: &[Option<f64>], predicate: fn(GpuPassGroup) -> bool) -> Option<f64> {
    GpuPassGroup::ALL
        .iter()
        .zip(group_ms)
        .filter(|(group, _)| predicate(**group))
        .filter_map(|(_, ms)| *ms)
        .reduce(|a, b| a + b)
}

fn ms_text(ms: Option<f64>) -> String {
    match ms {
        Some(ms) => format!("{ms:.2} ms"),
        None => "-".to_string(),
    }
}

fn is_lighting(group: GpuPassGroup) -> bool {
    matches!(
        group,
        GpuPassGroup::BasePass | GpuPassGroup::DiffuseIndirect | GpuPassGroup::Lights
    )
}

fn pass_rows(parent: &web_sys::Element, name: &str, row_id: fn(usize) -> String) {
    let mut lighting = None;
    for (index, group) in GpuPassGroup::ALL.iter().enumerate() {
        if is_lighting(*group) {
            let lighting = lighting.get_or_insert_with(|| {
                accordion(parent, &format!("performance-{name}-lighting"), "Lighting", true)
            });
            append(lighting, &stat_row(group.label(), &row_id(index), 0));
        } else if !is_postprocess(*group) && !is_overlay(*group) {
            append(parent, &stat_row(group.label(), &row_id(index), 0));
        }
    }
    if lighting.is_some() {
        accordion_value(parent, &format!("performance-{name}-lighting"));
    }

    let postprocess = accordion(
        parent,
        &format!("performance-{name}-postprocess"),
        "Postprocess",
        true,
    );
    for (index, group) in GpuPassGroup::ALL.iter().enumerate() {
        if is_postprocess(*group) {
            append(&postprocess, &stat_row(group.label(), &row_id(index), 0));
        }
    }
    accordion_value(parent, &format!("performance-{name}-postprocess"));

    let overlay = accordion(parent, &format!("performance-{name}-overlay"), "Overlay", true);
    for (index, group) in GpuPassGroup::ALL.iter().enumerate() {
        if is_overlay(*group) {
            append(&overlay, &stat_row(group.label(), &row_id(index), 0));
        }
    }
    accordion_value(parent, &format!("performance-{name}-overlay"));
}
