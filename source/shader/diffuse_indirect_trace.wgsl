const DIFFUSE_RAY_COUNT : i32 = 8;
const DIFFUSE_STEPS     : i32 = 16;
// Allowed depth gap behind the surface, relative to the distance from the camera
const DIFFUSE_THICKNESS : f32 = 0.05;
// Moves the ray origin off the surface to avoid hitting itself
const SURFACE_OFFSET    : f32 = 0.02;

// gbuffers (normal, depth, albedo, metallic) and the previous frame's lit scene (history)
@group(0) @binding(0) var t_normal   : texture_2d<f32>;
@group(0) @binding(1) var t_depth    : texture_depth_2d;
@group(0) @binding(2) var t_albedo   : texture_2d<f32>;
@group(0) @binding(3) var t_metallic : texture_2d<f32>;
@group(0) @binding(4) var t_history  : texture_2d<f32>;
@group(1) @binding(0) var<uniform>  u_indirect : IndirectUniform;

struct TraceOutput
{
    @location(0) indirect : vec4<f32>,
    // x: view distance of the surface, y: 1 sample
    @location(1) distance_and_count : vec2<f32>,
}

fn gbuffer_position(texel : vec2<i32>) -> vec3<f32>
{
    let dims : vec2<f32> = vec2<f32>(textureDimensions(t_depth));
    return reconstruct_world_position(vec2<f32>(texel) + 0.5, dims, textureLoad(t_depth, texel, 0), u_indirect.inv_view_projection);
}

fn interleaved_gradient_noise(p : vec2<f32>) -> f32
{
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// Marches a ray in screen space. Returns rgb = previous frame radiance at the hit, a = 1.0 on hit.
fn trace_diffuse(origin : vec3<f32>, dir : vec3<f32>) -> vec4<f32>
{
    let eye          : vec3<f32> = u_indirect.camera_position.xyz;
    let dims         : vec2<i32> = vec2<i32>(textureDimensions(t_depth));
    let step_length  : f32       = u_indirect.params.x / f32(DIFFUSE_STEPS);

    for (var i : i32 = 1; i <= DIFFUSE_STEPS; i = i + 1)
    {
        let p    : vec3<f32> = origin + dir * (f32(i) * step_length);
        let clip : vec4<f32> = u_indirect.view_projection * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0)
        {
            break;
        }

        let ndc : vec3<f32> = clip.xyz / clip.w;
        if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z < 0.0 || ndc.z > 1.0)
        {
            break;
        }

        let uv    : vec2<f32> = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        let texel : vec2<i32> = clamp(vec2<i32>(floor(uv * vec2<f32>(dims))), vec2<i32>(0), dims - vec2<i32>(1));

        if (textureLoad(t_depth, texel, 0) >= 1.0)
        {
            continue;
        }

        let stored_distance : f32 = length(gbuffer_position(texel) - eye);
        let ray_distance    : f32 = length(p - eye);
        if (ray_distance > stored_distance && ray_distance - stored_distance < DIFFUSE_THICKNESS * stored_distance)
        {
            return vec4<f32>(textureLoad(t_history, texel, 0).rgb, 1.0);
        }
    }

    return vec4<f32>(0.0);
}

// Output: indirect = diffuse indirect light (albedo * (1 - metallic) * mean radiance) and ambient occlusion,
// distance = view distance of the surface (0 for background)
@fragment
fn fs_main( @builtin(position) coord : vec4f ) -> TraceOutput
{
    let texel = vec2<i32>(floor(coord.xy));
    if (textureLoad(t_depth, texel, 0) >= 1.0)
    {
        return TraceOutput(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec2<f32>(0.0, 0.0));
    }

    let eye       : vec3<f32> = u_indirect.camera_position.xyz;
    let position  : vec3<f32> = gbuffer_position(texel);
    let normal    : vec3<f32> = safe_normalize(textureLoad(t_normal, texel, 0).xyz * 2.0 - 1.0);
    let albedo    : vec3<f32> = textureLoad(t_albedo, texel, 0).rgb;
    let metallic  : f32       = clamp(textureLoad(t_metallic, texel, 0).b, 0.0, 1.0);

    let up        : vec3<f32> = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.z) < 0.999);
    let tangent   : vec3<f32> = safe_normalize(cross(up, normal));
    let bitangent : vec3<f32> = cross(normal, tangent);
    let origin    : vec3<f32> = position + normal * SURFACE_OFFSET;

    // Stratified samples: each ray takes its own stratum of the radial variable. The per-pixel noise and a
    // per-frame rotation decorrelate pixels and frames, so the temporal accumulation covers the hemisphere.
    let frame       : f32 = u_indirect.params.w;
    let pixel_noise : f32 = fract(interleaved_gradient_noise(vec2<f32>(texel)) + fract(frame * 0.7548776662));
    let azimuth_noise : f32 = fract(pixel_noise * 7.31 + 0.5);

    var radiance_sum : vec3<f32> = vec3<f32>(0.0);
    var hit_count    : f32       = 0.0;
    for (var i : i32 = 0; i < DIFFUSE_RAY_COUNT; i = i + 1)
    {
        let fi : f32 = f32(i);
        let u1 : f32 = fract((fi + 0.5) / f32(DIFFUSE_RAY_COUNT) + pixel_noise);
        let u2 : f32 = fract(fi * 0.618034 + azimuth_noise);

        // cosine-weighted hemisphere sample around the normal
        let r     : f32 = sqrt(u1);
        let phi   : f32 = 6.2831853 * u2;
        let local : vec3<f32> = vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(1.0 - u1, 0.0)));
        let dir   : vec3<f32> = tangent * local.x + bitangent * local.y + normal * local.z;

        let hit : vec4<f32> = trace_diffuse(origin, dir);
        radiance_sum += hit.rgb;
        hit_count += hit.a;
    }

    // With cosine-weighted samples, albedo / pi * integral(L cos) reduces to albedo * mean(L)
    let diffuse : vec3<f32> = (1.0 - metallic) * albedo * (radiance_sum / f32(DIFFUSE_RAY_COUNT)) * u_indirect.params.z;
    let occlusion : f32 = hit_count / f32(DIFFUSE_RAY_COUNT);
    let ambient_occlusion : f32 = mix(1.0, 1.0 - occlusion, u_indirect.params.y);
    return TraceOutput(vec4<f32>(diffuse, ambient_occlusion), vec2<f32>(length(position - eye), 1.0));
}
