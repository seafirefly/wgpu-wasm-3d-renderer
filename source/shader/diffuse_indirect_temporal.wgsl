// raw trace output of this frame (distance_and_count: x = view distance, y = 1 sample)
@group(0) @binding(0) var t_raw          : texture_2d<f32>;
@group(0) @binding(1) var t_raw_distance : texture_2d<f32>;
// accumulated result and its distance_and_count (x = view distance, y = sample count) from the previous frame
@group(0) @binding(2) var t_history          : texture_2d<f32>;
@group(0) @binding(3) var t_history_distance : texture_2d<f32>;
@group(0) @binding(4) var t_depth        : texture_depth_2d;
@group(1) @binding(0) var<uniform>  u_indirect : IndirectUniform;

struct TemporalOutput
{
    @location(0) indirect : vec4<f32>,
    @location(1) distance_and_count : vec2<f32>,
}

fn gbuffer_position(texel : vec2<i32>) -> vec3<f32>
{
    let dims : vec2<f32> = vec2<f32>(textureDimensions(t_depth));
    return reconstruct_world_position(vec2<f32>(texel) + 0.5, dims, textureLoad(t_depth, texel, 0), u_indirect.inv_view_projection);
}

// Cumulative moving average of the trace over the reprojected history.
// The history is rejected when its surface does not match (disocclusion) and clamped to the current 3x3 color range.
// The sample count grows by one per frame up to the maximum, so the weight of the current frame is 1 / count.
@fragment
fn fs_main( @builtin(position) coord : vec4f ) -> TemporalOutput
{
    let texel = vec2<i32>(floor(coord.xy));
    let raw = textureLoad(t_raw, texel, 0);
    let distance = textureLoad(t_raw_distance, texel, 0).x;
    if (distance <= 0.0)
    {
        return TemporalOutput(raw, vec2<f32>(0.0, 0.0));
    }

    let first_sample = TemporalOutput(raw, vec2<f32>(distance, 1.0));
    if (u_indirect.history.x < 0.5)
    {
        return first_sample;
    }

    let dims  : vec2<i32> = vec2<i32>(textureDimensions(t_raw));
    let position : vec3<f32> = gbuffer_position(texel);

    let prev_clip : vec4<f32> = u_indirect.prev_view_projection * vec4<f32>(position, 1.0);
    if (prev_clip.w <= 0.0)
    {
        return first_sample;
    }
    let prev_ndc : vec3<f32> = prev_clip.xyz / prev_clip.w;
    let prev_uv  : vec2<f32> = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
    let prev_pixel : vec2<f32> = floor(prev_uv * vec2<f32>(dims));
    if (any(prev_pixel < vec2<f32>(0.0)) || any(prev_pixel >= vec2<f32>(dims)))
    {
        return first_sample;
    }
    let prev_texel = vec2<i32>(prev_pixel);

    // the same surface is at the same distance from the previous camera; otherwise it was disoccluded
    let expected_distance : f32 = length(u_indirect.prev_camera_position.xyz - position);
    let previous_info : vec2<f32> = textureLoad(t_history_distance, prev_texel, 0).xy;
    if (abs(previous_info.x - expected_distance) > u_indirect.history.z * expected_distance)
    {
        return first_sample;
    }

    var min_color : vec4<f32> = raw;
    var max_color : vec4<f32> = raw;
    for (var dy : i32 = -1; dy <= 1; dy = dy + 1)
    {
        for (var dx : i32 = -1; dx <= 1; dx = dx + 1)
        {
            let q : vec2<i32> = clamp(texel + vec2<i32>(dx, dy), vec2<i32>(0), dims - vec2<i32>(1));
            let c : vec4<f32> = textureLoad(t_raw, q, 0);
            min_color = min(min_color, c);
            max_color = max(max_color, c);
        }
    }
    let previous : vec4<f32> = clamp(textureLoad(t_history, prev_texel, 0), min_color, max_color);

    let count : f32 = min(previous_info.y + 1.0, u_indirect.history.y);
    return TemporalOutput(mix(previous, raw, 1.0 / count), vec2<f32>(distance, count));
}
