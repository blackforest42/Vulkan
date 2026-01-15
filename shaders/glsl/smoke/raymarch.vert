#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 color;

// out
layout (location = 0) out vec3 outUVW;

layout (binding = 0) uniform UBOView
{
	mat4 model;
	mat4 cameraView;
	mat4 perspective;
    vec3 cameraPos;
    vec2 screenRes;
    float time;
} ubo;


void main() {
    gl_Position = ubo.perspective * ubo.cameraView * ubo.model * vec4(inPos, 1.);
    // Use vertex positions [-0.5, 0.5]to create UV positions [0, 1]
    outUVW = (inPos + 0.5f);
}

