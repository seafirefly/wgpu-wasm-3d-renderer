struct EnvUniform
{
    view_projection : mat4x4<f32>,
    camera_position : vec4<f32>,
    // x: 1.0 when screen space reflection is enabled, y: 1.0 when IBL is available, z: 1.0 when IBL diffuse is enabled, w: 1.0 when IBL specular is enabled
    ssr_params      : vec4<f32>,
    // uniform ambient light (rgb, intensity in w)
    ambient_light   : vec4<f32>,
}

const SSR_STEPS        : i32 = 48;
const SSR_MAX_DISTANCE : f32 = 20.0;
// Allowed depth gap behind the surface, relative to the distance from the camera
const SSR_THICKNESS    : f32 = 0.05;

// deferred gbuffers (same layout as the deferred lighting pass)
@group(0) @binding(0) var t_positon : texture_2d<f32>;
@group(0) @binding(1) var t_normal  : texture_2d<f32>;
@group(0) @binding(2) var t_depth   : texture_depth_2d;
@group(0) @binding(3) var t_albedo  : texture_2d<f32>;
@group(0) @binding(4) var t_metallic : texture_2d<f32>;
@group(1) @binding(0) var<uniform>  u_env : EnvUniform;
// IBL
@group(2) @binding(0) var t_irradiance : texture_cube<f32>;
@group(2) @binding(1) var t_prefilter  : texture_cube<f32>;
@group(2) @binding(2) var s_env        : sampler;
// lit scene color before the environment terms are added
@group(3) @binding(0) var t_scene      : texture_2d<f32>;

// Marches the reflection ray in screen space. Returns rgb = hit color, a = hit confidence.
fn screen_space_reflection(origin : vec3<f32>, dir : vec3<f32>) -> vec4<f32>
{
    let eye          : vec3<f32> = u_env.camera_position.xyz;
    let dims         : vec2<i32> = vec2<i32>(textureDimensions(t_scene));
    let step_length  : f32       = SSR_MAX_DISTANCE / f32(SSR_STEPS);

    for (var i : i32 = 1; i <= SSR_STEPS; i = i + 1)
    {
        let p    : vec3<f32> = origin + dir * (f32(i) * step_length);
        let clip : vec4<f32> = u_env.view_projection * vec4<f32>(p, 1.0);
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

        // The ray has passed behind the visible surface at this pixel: treat it as a hit when it is close enough
        let stored_distance : f32 = length(textureLoad(t_positon, texel, 0).xyz - eye);
        let ray_distance    : f32 = length(p - eye);
        if (ray_distance > stored_distance && ray_distance - stored_distance < SSR_THICKNESS * stored_distance)
        {
            let border : f32 = 1.0 - max(abs(ndc.x), abs(ndc.y));
            return vec4<f32>(textureLoad(t_scene, texel, 0).rgb, smoothstep(0.0, 0.1, border));
        }
    }

    return vec4<f32>(0.0);
}

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
    let texel = vec2<i32>(floor(coord.xy));
    let scene = textureLoad(t_scene, texel, 0);

    // background pixels already contain the sky
    if (textureLoad(t_depth, texel, 0) >= 1.0)
    {
        return scene;
    }

    let position  : vec3<f32> = textureLoad(t_positon, texel, 0).xyz;
    let normal    : vec3<f32> = safe_normalize(textureLoad(t_normal, texel, 0).xyz);
    let albedo    : vec3<f32> = textureLoad(t_albedo, texel, 0).rgb;
    let mr        : vec4<f32> = textureLoad(t_metallic, texel, 0);
    let roughness : f32       = clamp(mr.g, 0.045, 1.0);
    let metallic  : f32       = clamp(mr.b, 0.0, 1.0);
    let view      : vec3<f32> = safe_normalize(u_env.camera_position.xyz - position);
    let n_dot_v   : f32       = max(dot(normal, view), 0.0);

    let f0       : vec3<f32> = mix(vec3<f32>(0.04, 0.04, 0.04), albedo, vec3<f32>(metallic));
    let fresnel  : vec3<f32> = fresnel_schlick(n_dot_v, f0);
    let kd       : vec3<f32> = (vec3<f32>(1.0, 1.0, 1.0) - fresnel) * (1.0 - metallic);

    // Ambient term: uniform environment light, applied to every surface
    let ambient_irradiance : vec3<f32> = u_env.ambient_light.rgb * max(u_env.ambient_light.w, 0.0);
    let ambient : vec3<f32> = pbr_ambient_diffuse_lighting(normal, view, albedo, metallic, ambient_irradiance);

    // IBL terms are only added when an environment map is available
    if (u_env.ssr_params.y < 0.5)
    {
        return vec4<f32>(scene.rgb + ambient, scene.a);
    }

    // Diffuse environment: the irradiance map already stores the cosine-weighted average radiance
    let irradiance : vec3<f32> = textureSampleLevel(t_irradiance, s_env, normal, 0.0).rgb;
    let diffuse    : vec3<f32> = kd * albedo * irradiance * u_env.ssr_params.z;

    // Specular environment: screen space reflection first, prefiltered environment as the fallback
    let reflect_dir : vec3<f32> = reflect(-view, normal);
    var radiance    : vec3<f32> = textureSampleLevel(t_prefilter, s_env, reflect_dir, roughness * 4.0).rgb;
    if (u_env.ssr_params.x > 0.5)
    {
        let ssr        : vec4<f32> = screen_space_reflection(position, reflect_dir);
        let ssr_weight : f32       = ssr.a * smoothstep(0.5, 0.1, roughness);
        radiance = mix(radiance, ssr.rgb, ssr_weight);
    }

    let env_brdf  : vec2<f32> = env_brdf_approx(roughness, n_dot_v);
    let specular  : vec3<f32> = radiance * (f0 * env_brdf.x + vec3<f32>(env_brdf.y)) * u_env.ssr_params.w;

    return vec4<f32>(scene.rgb + ambient + diffuse + specular, scene.a);
}
