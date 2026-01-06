#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec3 inPos;

// out
layout (location = 0) out vec3 outPos;

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
    outPos = inPos;
    gl_Position = ubo.perspective * ubo.cameraView * ubo.model * vec4(inPos, 1.);
}

