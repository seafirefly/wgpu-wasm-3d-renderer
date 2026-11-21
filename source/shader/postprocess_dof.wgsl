struct VertexOutput
{
    @builtin(position) position : vec4<f32>,
    @location(0)       uv       : vec2<f32>
};

struct DofUniform
{
    focus_distance : f32,
    aperture       : f32,
    near           : f32,
    far            : f32,
}

@group(0) @binding(0) var           t_scene  : texture_2d<f32>;
@group(0) @binding(1) var           s_scene  : sampler;
@group(0) @binding(2) var           t_depth  : texture_depth_2d;
@group(1) @binding(0) var<uniform>  u_dof    : DofUniform;

const TAP_COUNT       : i32 = 32;
const GOLDEN_ANGLE    : f32 = 2.39996323;
const MAX_COC_FRACTION : f32 = 0.05;

fn linear_depth(depth : f32) -> f32 {
    return u_dof.near * u_dof.far / (u_dof.far - depth * (u_dof.far - u_dof.near));
}

// Radius of the circle of confusion in pixels. The aperture is a fraction of the screen height.
fn coc_radius(view_depth : f32, screen_height : f32) -> f32 {
    let coc = u_dof.aperture * abs(view_depth - u_dof.focus_distance) / view_depth;
    return min(coc * screen_height, MAX_COC_FRACTION * screen_height);
}

@vertex
fn vs_main( @builtin(vertex_index) vertex_index : u32 ) -> VertexOutput
{
    var out: VertexOutput;

    let x = f32(i32(vertex_index & 1u) << 2u) - 1.0;
    let y = f32(i32(vertex_index & 2u) << 1u) - 1.0;

    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv       = vec2f(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

// Gather blur: each tap contributes only if its own circle of confusion reaches the current pixel,
// which keeps sharp foreground from bleeding into blurred background.
@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4f
{
    let dims = textureDimensions(t_scene);
    let screen_height = f32(dims.y);
    let max_pixel = vec2<i32>(dims) - vec2<i32>(1);
    let center_pixel = vec2<i32>(in.position.xy);

    let center_color = textureLoad(t_scene, center_pixel, 0).rgb;
    let center_depth = linear_depth(textureLoad(t_depth, center_pixel, 0));
    let center_radius = coc_radius(center_depth, screen_height);

    if (center_radius < 0.5) {
        return vec4f(center_color, 1.0);
    }

    // Per-pixel rotation of the sampling pattern turns banding into fine noise
    let jitter = fract(52.9829189 * fract(dot(in.position.xy, vec2f(0.06711056, 0.00583715)))) * 6.2831853;

    var color_sum  = center_color;
    var weight_sum = 1.0;

    for (var i = 0; i < TAP_COUNT; i++) {
        let fi = f32(i);
        let r = sqrt((fi + 0.5) / f32(TAP_COUNT)) * center_radius;
        let theta = fi * GOLDEN_ANGLE + jitter;
        let offset = vec2f(cos(theta), sin(theta)) * r;

        let tap_pixel = clamp(vec2<i32>(in.position.xy + offset), vec2<i32>(0), max_pixel);
        let tap_radius = coc_radius(linear_depth(textureLoad(t_depth, tap_pixel, 0)), screen_height);
        let weight = clamp(tap_radius - r + 1.0, 0.0, 1.0);

        let tap_uv = (vec2f(tap_pixel) + vec2f(0.5)) / vec2f(dims);
        color_sum  += textureSampleLevel(t_scene, s_scene, tap_uv, 0.0).rgb * weight;
        weight_sum += weight;
    }

    return vec4f(color_sum / weight_sum, 1.0);
}
