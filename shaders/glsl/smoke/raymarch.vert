#version 450
#extension GL_EXT_debug_printf : enable

// in

// out
layout (location = 0) out vec2 outUV;

layout (binding = 0) uniform UBO
{
	mat4 cameraView;
    vec3 cameraPos;
    vec2 screenRes;
    float time;
} ubo;

vec2 positions[6] = vec2[](
    // bottom left tri
    vec2(-1, -1),
    vec2(-1, 1),
    vec2(1, 1),

    // top right tri
    vec2(1, 1),
    vec2(1, -1),
    vec2(-1, -1)
);

void main() {
    outUV = (positions[gl_VertexIndex].xy + 1.0) * 0.5;
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}


