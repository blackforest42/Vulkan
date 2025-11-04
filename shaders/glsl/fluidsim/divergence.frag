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

layout (binding = 1) uniform sampler2D vectorField;

void main() {
	vec2 dudv = 1 / ubo.bufferResolution;
	float du = dudv.x;
	float dv = dudv.y;

	vec4 wL = texture(vectorField, inUV - vec2(du, 0));
	vec4 wR = texture(vectorField, inUV + vec2(du, 0));
	vec4 wB = texture(vectorField, inUV - vec2(0, dv));
	vec4 wT = texture(vectorField, inUV + vec2(0, dv));
	outFragColor = vec4(0.5f * ((wR.x - wL.x) + (wT.y - wB.y)));
}