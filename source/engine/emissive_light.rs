use crate::engine::scene::Scene;

// Emissive triangles are sampled as point area lights. Keep in sync with MAX_EMISSIVE_LIGHTS in the shaders.
pub const MAX_EMISSIVE_LIGHTS: usize = 4;
pub const POINT_SHADOW_MAP_SIZE: u32 = 2048;
const POINT_SHADOW_FOV_DEGREES: f32 = 160.0;
const POINT_SHADOW_NEAR: f32 = 0.01;
const POINT_SHADOW_FAR: f32 = 100.0;
// Floats per light in the GPU layout: position+area, radiance, normal+shadow layer, view_projection
const GPU_LIGHT_FLOATS: usize = 28;

#[derive(Clone, Copy)]
pub struct EmissiveLight {
    // render space (Z-up), world units
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
    pub radiance: glam::Vec3,
    pub area: f32,
}

/// Builds the light list from emissive triangles (triangle centroid, area, averaged vertex normal).
/// Keeps the largest MAX_EMISSIVE_LIGHTS triangles.
pub fn collect_emissive_lights(scene: &Scene) -> Vec<EmissiveLight> {
    let mut lights: Vec<EmissiveLight> = Vec::new();

    for object in scene.objects.iter() {
        let Some(source_mesh) = object.source_mesh.as_ref() else {
            continue;
        };
        let mesh = source_mesh.borrow();
        let Some(material) = mesh
            .material
            .and_then(|index| scene.materials.get(index as usize))
        else {
            continue;
        };
        let radiance = glam::Vec3::from_array(material.emissive_factor);
        if radiance.max_element() <= 0.0 {
            continue;
        }

        let mut model = glam::Mat4::from_cols_array_2d(&object.world_transform);
        if scene.parameters.is_convert_y_to_z {
            let y_to_z = glam::Mat4::from_axis_angle(glam::Vec3::X, std::f32::consts::PI / 2.0);
            model = y_to_z * model;
        }

        for triangle in mesh.indices.chunks_exact(3) {
            let vertices = [
                &mesh.vertices[triangle[0] as usize],
                &mesh.vertices[triangle[1] as usize],
                &mesh.vertices[triangle[2] as usize],
            ];
            let positions = vertices.map(|v| model.transform_point3(glam::Vec3::new(v._pos[0], v._pos[1], v._pos[2])));
            let area = 0.5 * (positions[1] - positions[0]).cross(positions[2] - positions[0]).length();
            let normal_sum: glam::Vec3 = vertices
                .iter()
                .map(|v| glam::Vec3::from_array(v._normal))
                .sum();
            let normal = model.transform_vector3(normal_sum).normalize_or_zero();
            if area <= 0.0 || normal == glam::Vec3::ZERO {
                continue;
            }

            lights.push(EmissiveLight {
                position: (positions[0] + positions[1] + positions[2]) / 3.0,
                normal,
                radiance,
                area,
            });
        }
    }

    lights.sort_by(|a, b| b.area.total_cmp(&a.area));
    lights.truncate(MAX_EMISSIVE_LIGHTS);
    lights
}

/// View-projection used for the light's shadow map. The camera looks along the emitting normal.
pub fn point_light_view_projection(light: &EmissiveLight) -> glam::Mat4 {
    let up = if light.normal.z.abs() > 0.99 {
        glam::Vec3::Y
    } else {
        glam::Vec3::Z
    };
    let view = glam::Mat4::look_to_rh(light.position, light.normal, up);
    let projection = glam::Mat4::perspective_rh(
        POINT_SHADOW_FOV_DEGREES.to_radians(),
        1.0,
        POINT_SHADOW_NEAR,
        POINT_SHADOW_FAR,
    );
    projection * view
}

/// GPU layout of `count` (vec4) followed by MAX_EMISSIVE_LIGHTS light structs. Matches `EmissiveLight` in pbr_common.wgsl.
pub fn emissive_light_uniform_values(lights: &[EmissiveLight]) -> Vec<f32> {
    let count = lights.len().min(MAX_EMISSIVE_LIGHTS);
    let mut values = vec![count as f32, 0.0, 0.0, 0.0];
    for index in 0..MAX_EMISSIVE_LIGHTS {
        if index < count {
            let light = &lights[index];
            values.extend_from_slice(&[light.position.x, light.position.y, light.position.z, light.area]);
            values.extend_from_slice(&[light.radiance.x, light.radiance.y, light.radiance.z, 0.0]);
            values.extend_from_slice(&[light.normal.x, light.normal.y, light.normal.z, index as f32]);
            values.extend_from_slice(&point_light_view_projection(light).to_cols_array());
        } else {
            values.extend_from_slice(&[0.0; GPU_LIGHT_FLOATS]);
        }
    }
    values
}
