struct Uniform
{
    directional_light     : vec4<f32>,
    ambient_light         : vec4<f32>,
    camera_position       : vec4<f32>,
    light_view_projection : mat4x4<f32>,
    inv_view_projection   : mat4x4<f32>,
    debug_params          : vec4<f32>,
    emissive_light_count  : vec4<f32>,
    emissive_lights       : array<EmissiveLight, MAX_EMISSIVE_LIGHTS>,
}

// basic differed (binding 0 was the position target, now rebuilt from depth)
@group(0) @binding(1) var t_normal        : texture_2d<f32>;
@group(0) @binding(2) var t_depth         : texture_depth_2d;
@group(0) @binding(3) var t_albedo        : texture_2d<f32>;
@group(0) @binding(4) var t_metallic      : texture_2d<f32>;
@group(1) @binding(0) var<uniform>  u_differd       : Uniform;
@group(2) @binding(0) var           t_shadow_map    : texture_depth_2d;
@group(2) @binding(1) var           s_shadow_map    : sampler_comparison;
@group(2) @binding(2) var           t_point_shadow_maps : texture_depth_2d_array;
// diffuse indirect light (rgb, already weighted by albedo) and screen space ambient occlusion (a)
@group(3) @binding(0) var t_diffuse_indirect : texture_2d<f32>;

// fragment helpers

struct GBufferTexel
{
  position           : vec3<f32>,
  normal             : vec3<f32>,
  depth              : f32,
  albedo             : vec4<f32>,
  metallic_roughness : vec4<f32>,
  is_emissive        : bool,
}

struct Surface
{
  position  : vec3<f32>,
  normal    : vec3<f32>,
  albedo    : vec3<f32>,
  roughness : f32,
  metallic  : f32,
  view      : vec3<f32>,
}

fn load_gbuffer(coord : vec4<f32>) -> GBufferTexel
{
  let texel = vec2<i32>(floor(coord.xy));
  let depth = textureLoad(t_depth, texel, 0);
  let normal_texel = textureLoad(t_normal, texel, 0);
  let metallic_texel = textureLoad(t_metallic, texel, 0);
  let dims = vec2<f32>(textureDimensions(t_depth));
  return GBufferTexel(
    reconstruct_world_position(coord.xy, dims, depth, u_differd.inv_view_projection),
    normal_texel.xyz * 2.0 - 1.0,
    depth,
    textureLoad(t_albedo, texel, 0),
    metallic_texel,
    metallic_texel.a > 0.5,
  );
}

fn make_surface(g : GBufferTexel) -> Surface
{
  return Surface(
    g.position,
    safe_normalize(g.normal),
    g.albedo.rgb,
    clamp(g.metallic_roughness.g, 0.045, 1.0),
    clamp(g.metallic_roughness.b, 0.0, 1.0),
    safe_normalize(u_differd.camera_position.xyz - g.position),
  );
}

fn light_direction() -> vec3<f32>
{
  return safe_normalize(-u_differd.directional_light.xyz);
}

fn light_radiance() -> f32
{
  return max(u_differd.directional_light.w, 0.0);
}

fn shadow_visibility_for(s : Surface, light : vec3<f32>) -> f32
{
  return sample_directional_shadow(t_shadow_map, s_shadow_map, u_differd.light_view_projection, s.position, s.normal, light);
}

// direct light multiplied by radiance and shadow visibility
fn direct_radiance(s : Surface) -> vec3<f32>
{
  let light = light_direction();
  let shadow_visibility = shadow_visibility_for(s, light);
  let lighting = pbr_direct_lighting(s.normal, s.view, light, s.albedo, s.roughness, s.metallic);
  return lighting.direct * light_radiance() * shadow_visibility;
}

// emissive triangles as area lights, each with its own shadow map layer
fn emissive_radiance(s : Surface) -> vec3<f32>
{
  var total = vec3<f32>(0.0, 0.0, 0.0);
  let count = u32(u_differd.emissive_light_count.x);
  for (var i = 0u; i < count; i++)
  {
    let light = u_differd.emissive_lights[i];
    let direct = emissive_light_direct(light, s.position, s.normal, s.view, s.albedo, s.roughness, s.metallic);
    let visibility = sample_point_light_shadow(t_point_shadow_maps, s_shadow_map, light, s.position, s.normal);
    total += direct * visibility;
  }
  return total;
}

// render functions

@vertex
fn vs_main( @builtin(vertex_index) VertexIndex : u32 ) -> @builtin(position) vec4f 
{
  const pos = array(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
  );

  return vec4f(pos[VertexIndex], 0.0, 1.0);
}

@fragment
fn fs_main( @builtin(position) coord : vec4f ) -> @location(0) vec4f
{
  let g = load_gbuffer(coord);
  if (g.depth >= 1.0)
  {
    discard;
  }

  let s = make_surface(g);

  // emissive surfaces only show their own emission, so they are not lit by the emissive lights
  // (emissive radiance itself is already in SceneColor)
  let area_radiance = emissive_radiance(s) * select(1.0, 0.0, g.is_emissive);

  // diffuse indirect light, occluded by the screen space ambient occlusion in alpha
  let indirect = textureLoad(t_diffuse_indirect, vec2<i32>(floor(coord.xy)), 0);
  let indirect_radiance = indirect.rgb * indirect.a * select(1.0, 0.0, g.is_emissive);

  return vec4<f32>(direct_radiance(s) + area_radiance + indirect_radiance, g.albedo.a);
}

@fragment
fn fs_debug_main( @builtin(position) coord : vec4f ) -> @location(0) vec4f
{
  let g = load_gbuffer(coord);
  let s = make_surface(g);
  let mode = u32(u_differd.debug_params.x);

  switch (mode) {
    case 1u: {
      return vec4<f32>(g.normal * 0.5 + 0.5, 1.0);
    }
    case 3u: {
      return g.albedo;
    }
    case 4u: {
      return vec4<f32>(vec3<f32>(s.metallic), 1.0);
    }
    case 5u: {
      return vec4<f32>(vec3<f32>(s.roughness), 1.0);
    }
    case 6u: {
      let light = light_direction();
      let lighting = pbr_direct_lighting(s.normal, s.view, light, s.albedo, s.roughness, s.metallic);
      let shadow_visibility = shadow_visibility_for(s, light);
      return vec4<f32>(lighting.specular * light_radiance() * shadow_visibility, 1.0);
    }
    default: {
      return vec4<f32>((1.0 - g.depth) * 50.0, 0.0, 0.0, 1.0);
    }
  }
}
