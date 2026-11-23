// accumulated indirect light, view distance, and normal of the gbuffer
@group(0) @binding(0) var t_accumulated : texture_2d<f32>;
@group(0) @binding(1) var t_distance    : texture_2d<f32>;
@group(0) @binding(2) var t_normal      : texture_2d<f32>;
@group(1) @binding(0) var<uniform>  u_indirect : IndirectUniform;

const DENOISE_RADIUS       : i32 = 2;
const DENOISE_NORMAL_POWER : f32 = 32.0;

// Edge-aware blur: neighbors are weighted by normal and view distance similarity
@fragment
fn fs_main( @builtin(position) coord : vec4f ) -> @location(0) vec4<f32>
{
    let texel = vec2<i32>(floor(coord.xy));
    let dims  : vec2<i32> = vec2<i32>(textureDimensions(t_accumulated));
    let center = textureLoad(t_accumulated, texel, 0);
    let center_distance : f32 = textureLoad(t_distance, texel, 0).r;
    if (center_distance <= 0.0)
    {
        return center;
    }
    let center_normal : vec3<f32> = safe_normalize(textureLoad(t_normal, texel, 0).xyz * 2.0 - 1.0);

    var sum        : vec4<f32> = vec4<f32>(0.0);
    var weight_sum : f32       = 0.0;
    for (var dy : i32 = -DENOISE_RADIUS; dy <= DENOISE_RADIUS; dy = dy + 1)
    {
        for (var dx : i32 = -DENOISE_RADIUS; dx <= DENOISE_RADIUS; dx = dx + 1)
        {
            let q : vec2<i32> = clamp(texel + vec2<i32>(dx, dy), vec2<i32>(0), dims - vec2<i32>(1));
            let distance : f32 = textureLoad(t_distance, q, 0).r;
            if (distance <= 0.0)
            {
                continue;
            }

            let normal : vec3<f32> = safe_normalize(textureLoad(t_normal, q, 0).xyz * 2.0 - 1.0);
            let weight_normal  : f32 = pow(max(dot(center_normal, normal), 0.0), DENOISE_NORMAL_POWER);
            let weight_depth   : f32 = exp(-abs(distance - center_distance) / (center_distance * u_indirect.history.z));
            let weight_spatial : f32 = exp(-f32(dx * dx + dy * dy) / 8.0);
            let weight : f32 = weight_normal * weight_depth * weight_spatial;

            sum += textureLoad(t_accumulated, q, 0) * weight;
            weight_sum += weight;
        }
    }

    if (weight_sum <= 0.0001)
    {
        return center;
    }
    return sum / weight_sum;
}
