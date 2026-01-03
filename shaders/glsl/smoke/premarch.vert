#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec3 inPos;

// out
layout (location = 0) out vec2 outUV;

layout (binding = 0) uniform UBO
{
	mat4 model;
	mat4 view;
	mat4 perspective;
	vec3 cameraPos;
	vec2 screenRes;
	int enableFrontMarch;
} ubo;


void main() {
    gl_Position = ubo.perspective * ubo.view * ubo.model * vec4(inPos, 1.);
}


