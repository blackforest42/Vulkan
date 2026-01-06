#version 450
#extension GL_EXT_debug_printf : enable

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inColor;

layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBO
{
	mat4 model;
	mat4 view;
	mat4 perspective;
	vec3 cameraPos;
    vec2 screenRes;
} ubo;


void main() {
	vec3 result = normalize(inPos - ubo.cameraPos);
	outFragColor.rgb = result;
}
