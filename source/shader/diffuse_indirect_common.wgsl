struct IndirectUniform
{
    view_projection      : mat4x4<f32>,
    inv_view_projection  : mat4x4<f32>,
    camera_position      : vec4<f32>,
    prev_view_projection : mat4x4<f32>,
    prev_camera_position : vec4<f32>,
    // x: maximum trace distance (world units), y: AO strength, z: GI intensity, w: frame index
    params               : vec4<f32>,
    // x: 1.0 when the temporal history is valid, y: maximum sample count of the temporal accumulation, z: relative depth tolerance
    history              : vec4<f32>,
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
