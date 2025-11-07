#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBO
{
    vec2 bufferResolution;
} ubo;

const float PI = 3.14159265358979323846;

void main() {
	float x = inUV.x;
	float y = inUV.y;

	outFragColor.r = sin(8.0f * PI * y);
	outFragColor.g = sin(8.0f * PI * x);
	return;
}