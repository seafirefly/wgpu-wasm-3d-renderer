// Shared PBR helpers. Concatenated in front of differed_pbr.wgsl and forward_pbr.wgsl.

const PI : f32 = radians(180.0);
const GGX_D_EPSILON : f32 = 0.0000001;
const GGX_VIS_EPSILON : f32 = 0.0000001;
const SHADOW_FILTER_RADIUS : f32 = 2.0;

fn distribution_GGX(normal : vec3<f32>, half : vec3<f32>, roughness : f32) -> f32
{
  let alpha    : f32 = roughness * roughness;
  let alpha2   : f32 = alpha * alpha;
  let n_dot_h  : f32 = max(dot(normal, half), 0.0);
  let n_dot_h2 : f32 = n_dot_h * n_dot_h;

  let nom   : f32    = alpha2;
  var denom : f32    = (n_dot_h2 * (alpha2 - 1.0) + 1.0);
  denom              = PI * denom * denom;

  return nom / max(denom, GGX_D_EPSILON);
}

// Smith height-correlated G2 folded into 1/(4*NdotL*NdotV) — matches Blender Principled BSDF
fn visibility_GGX(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32
{
  let a  = roughness * roughness;
  let a2 = a * a;
  let lambda_v = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
  let lambda_l = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
  return 0.5 / max(lambda_v + lambda_l, GGX_VIS_EPSILON);
}

fn fresnel_schlick(cos_theta : f32, f0 : vec3<f32>) -> vec3<f32>
{
  return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

fn env_brdf_approx(roughness : f32, n_dot_v : f32) -> vec2<f32>
{
  // Karis' analytic approximation of the split-sum BRDF LUT
  let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
  let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
  let r = roughness * c0 + c1;
  let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
  return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

fn safe_normalize(v : vec3<f32>) -> vec3<f32>
{
  let len2 = dot(v, v);
  if (len2 <= 0.000001) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }

  return v * inverseSqrt(len2);
}

// World position of a pixel rebuilt from its depth (WebGPU 0..1 range). pixel is in texel units, dims is the texture size.
fn reconstruct_world_position(pixel : vec2<f32>, dims : vec2<f32>, depth : f32, inv_view_projection : mat4x4<f32>) -> vec3<f32>
{
  let ndc   = vec4<f32>(pixel.x / dims.x * 2.0 - 1.0, 1.0 - pixel.y / dims.y * 2.0, depth, 1.0);
  let world = inv_view_projection * ndc;
  return world.xyz / world.w;
}

struct PbrLighting
{
  direct   : vec3<f32>,
  specular : vec3<f32>,
}

// light: direction towards the light. Returns BRDF * n·l (radiance not applied).
fn pbr_direct_lighting(
  normal : vec3<f32>,
  view : vec3<f32>,
  light : vec3<f32>,
  albedo : vec3<f32>,
  roughness : f32,
  metallic : f32,
) -> PbrLighting
{
  let n_dot_l : f32 = max(dot(normal, light), 0.0);
  let n_dot_v : f32 = max(dot(normal, view), 0.0);

  if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
    return PbrLighting(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0));
  }

  let half_vector : vec3<f32> = safe_normalize(view + light);
  let h_dot_v : f32 = max(dot(half_vector, view), 0.0);

  let base_reflectivity = vec3<f32>(0.04, 0.04, 0.04);
  let f0 = mix(base_reflectivity, albedo, vec3<f32>(metallic, metallic, metallic));
  let ndf = distribution_GGX(normal, half_vector, roughness);
  let vis = visibility_GGX(n_dot_l, n_dot_v, roughness);
  let fresnel = fresnel_schlick(h_dot_v, f0);

  let specular = ndf * vis * fresnel;

  let ks = fresnel;
  let kd = (vec3<f32>(1.0, 1.0, 1.0) - ks) * (1.0 - metallic);
  let diffuse = kd * albedo / PI;
  let specular_contrib = specular * n_dot_l;

  return PbrLighting((diffuse + specular) * n_dot_l, specular_contrib);
}

// Uniform ambient diffuse. Independent of the view direction so brightness does not change with the camera.
fn pbr_ambient_diffuse_lighting(
  albedo : vec3<f32>,
  metallic : f32,
  ambient_irradiance : vec3<f32>,
) -> vec3<f32>
{
  let kd = 1.0 - metallic;
  return kd * albedo * ambient_irradiance / PI;
}

// light_direction: direction towards the light. Returns 1.0 when fully lit.
fn sample_directional_shadow(
  t_shadow_map : texture_depth_2d,
  s_shadow_map : sampler_comparison,
  light_view_projection : mat4x4<f32>,
  world_position : vec3<f32>,
  normal : vec3<f32>,
  light_direction : vec3<f32>,
) -> f32
{
  let shadow_clip = light_view_projection * vec4<f32>(world_position, 1.0);
  let safe_w = max(abs(shadow_clip.w), 0.000001);
  let inv_w = select(-1.0 / safe_w, 1.0 / safe_w, shadow_clip.w >= 0.0);
  let shadow_ndc = shadow_clip.xyz * inv_w;
  let shadow_uv_raw = vec2<f32>(shadow_ndc.x * 0.5 + 0.5, shadow_ndc.y * -0.5 + 0.5);
  let shadow_uv = clamp(shadow_uv_raw, vec2<f32>(0.001, 0.001), vec2<f32>(0.999, 0.999));
  let in_front = select(0.0, 1.0, shadow_clip.w > 0.0);
  let inside_uv = select(0.0, 1.0,
    shadow_uv_raw.x > 0.0 && shadow_uv_raw.x < 1.0 && shadow_uv_raw.y > 0.0 && shadow_uv_raw.y < 1.0);
  let inside_depth = select(0.0, 1.0, shadow_ndc.z > 0.0 && shadow_ndc.z < 1.0);
  let valid_shadow = in_front * inside_uv * inside_depth;

  let slope_bias = max(0.002 * (1.0 - max(dot(normal, light_direction), 0.0)), 0.0005);
  let shadow_depth = clamp(shadow_ndc.z - slope_bias, 0.0, 1.0);
  let texel_size = 1.0 / max(vec2<f32>(textureDimensions(t_shadow_map)), vec2<f32>(1.0, 1.0));

  // 5x5 PCF kernel: outer product of binomial [1, 4, 6, 4, 1] (weights sum to 256)
  let kernel_1d = array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0);
  var shadow_sample = 0.0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let offset = vec2<f32>(f32(x), f32(y)) * SHADOW_FILTER_RADIUS * texel_size;
      let weight = kernel_1d[x + 2] * kernel_1d[y + 2];
      shadow_sample += textureSampleCompare(t_shadow_map, s_shadow_map, shadow_uv + offset, shadow_depth) * weight;
    }
  }
  shadow_sample /= 256.0;

  return 1.0 - valid_shadow + shadow_sample * valid_shadow;
}

const MAX_EMISSIVE_LIGHTS : u32 = 4u;

// Mirrors engine::emissive_light::EmissiveLight (28 floats per light)
struct EmissiveLight
{
  position_area   : vec4<f32>, // xyz: position, w: area
  radiance        : vec4<f32>, // rgb: radiance
  normal_layer    : vec4<f32>, // xyz: emitting normal, w: shadow map layer
  view_projection : mat4x4<f32>,
}

// Point-approximated emissive triangle. Returns BRDF * n·l * radiance * (area * cos_light / d²), shadow not applied.
fn emissive_light_direct(
  light : EmissiveLight,
  world_position : vec3<f32>,
  normal : vec3<f32>,
  view : vec3<f32>,
  albedo : vec3<f32>,
  roughness : f32,
  metallic : f32,
) -> vec3<f32>
{
  let to_light     : vec3<f32> = light.position_area.xyz - world_position;
  let distance2    : f32       = max(dot(to_light, to_light), 0.0004);
  let light_dir    : vec3<f32> = to_light * inverseSqrt(distance2);
  let emit_cos     : f32       = max(dot(light.normal_layer.xyz, -light_dir), 0.0);
  if (emit_cos <= 0.0) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }

  let irradiance_scale : f32 = light.position_area.w * emit_cos / distance2;
  let lighting = pbr_direct_lighting(normal, view, light_dir, albedo, roughness, metallic);
  return lighting.direct * light.radiance.rgb * irradiance_scale;
}

// Returns 1.0 when the point is visible from the emissive light. Samples are taken without early returns (uniform control flow).
fn sample_point_light_shadow(
  t_shadow_map : texture_depth_2d_array,
  s_shadow_map : sampler_comparison,
  light : EmissiveLight,
  world_position : vec3<f32>,
  normal : vec3<f32>,
) -> f32
{
  let offset_position : vec3<f32> = world_position + normal * 0.02;
  let shadow_clip = light.view_projection * vec4<f32>(offset_position, 1.0);
  let safe_w = max(abs(shadow_clip.w), 0.000001);
  let shadow_ndc = shadow_clip.xyz / safe_w;
  let shadow_uv_raw = vec2<f32>(shadow_ndc.x * 0.5 + 0.5, shadow_ndc.y * -0.5 + 0.5);
  let shadow_uv = clamp(shadow_uv_raw, vec2<f32>(0.001, 0.001), vec2<f32>(0.999, 0.999));
  let in_front = select(0.0, 1.0, shadow_clip.w > 0.0);
  let inside_uv = select(0.0, 1.0,
    shadow_uv_raw.x > 0.0 && shadow_uv_raw.x < 1.0 && shadow_uv_raw.y > 0.0 && shadow_uv_raw.y < 1.0);
  let inside_depth = select(0.0, 1.0, shadow_ndc.z > 0.0 && shadow_ndc.z < 1.0);
  let valid_shadow = in_front * inside_uv * inside_depth;

  let light_dir = safe_normalize(light.position_area.xyz - world_position);
  let slope_bias = 0.0002 + 0.0003 * (1.0 - max(dot(normal, light_dir), 0.0));
  let shadow_depth = clamp(shadow_ndc.z - slope_bias, 0.0, 1.0);
  let layer = i32(light.normal_layer.w);
  let texel_size = 1.0 / max(vec2<f32>(textureDimensions(t_shadow_map)), vec2<f32>(1.0, 1.0));

  // 5x5 PCF kernel, same as the directional shadow
  let kernel_1d = array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0);
  var shadow_sample = 0.0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let offset = vec2<f32>(f32(x), f32(y)) * SHADOW_FILTER_RADIUS * texel_size;
      let weight = kernel_1d[x + 2] * kernel_1d[y + 2];
      shadow_sample += textureSampleCompare(t_shadow_map, s_shadow_map, shadow_uv + offset, layer, shadow_depth) * weight;
    }
  }
  shadow_sample /= 256.0;

  return 1.0 - valid_shadow + shadow_sample * valid_shadow;
}
