#version 450

layout (location = 0) in vec3 inColor;

layout (location = 0) out vec4 outFragColor;


layout (binding = 0) uniform UBO
{
    vec2 screenRes;
} ubo;


void main() {
	outFragColor.rgb = inColor;
}

