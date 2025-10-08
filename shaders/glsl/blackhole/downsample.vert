#version 450

// in
layout(location = 0) in vec3 position;

vec2 positions[6] = vec2[](
    vec2(-1, -1),
    vec2(-1, 1),
    vec2(1, 1),

    vec2(1, 1),
    vec2(1, -1),
    vec2(-1, -1)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
