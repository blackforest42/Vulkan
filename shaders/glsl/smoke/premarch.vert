#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inColor;

// out
layout (location = 0) out vec3 outPos;
layout (location = 1) out vec3 outColor;

layout (binding = 0) uniform UBO
{
	mat4 model;
	mat4 view;
	mat4 perspective;
	vec3 cameraPos;
	vec2 screenRes;
} ubo;


void main() {
	outPos = inPos;
	outColor = inColor;
    gl_Position = ubo.perspective * ubo.view * ubo.model * vec4(inPos, 1.);
}


