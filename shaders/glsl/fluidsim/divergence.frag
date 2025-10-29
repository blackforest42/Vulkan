#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBO
{
	float halfRdx;
} ubo;

layout (binding = 1) uniform sampler2D vectorField;

void main() {
	vec2 coords = gl_FragCoord.xy;
	vec4 wL = texture(vectorField, coords + vec2(-1, 0));
	vec4 wR = texture(vectorField, coords + vec2(1, 0));
	vec4 wB = texture(vectorField, coords + vec2(0, -1));
	vec4 wT = texture(vectorField, coords + vec2(0, 1));
	outFragColor = vec4(ubo.halfRdx * ((wR.x - wL.x) + (wT.y - wB.y)));
}