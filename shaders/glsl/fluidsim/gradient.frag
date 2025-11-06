#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBO
{
    vec2 bufferResolution;
	float timestep;
} ubo;

layout (binding = 1) uniform sampler2D velocityFieldTex;
layout (binding = 2) uniform sampler2D pressureFieldTex;

void main() {
	vec2 dudv = 1.f / ubo.bufferResolution;
	float du = dudv.x;
	float dv = dudv.y;

	float pL = texture(pressureFieldTex, (inUV - vec2(du, 0))).x;
	float pR = texture(pressureFieldTex, (inUV + vec2(du, 0))).x;
	float pB = texture(pressureFieldTex, (inUV - vec2(0, dv))).x;
	float pT = texture(pressureFieldTex, (inUV + vec2(0, dv))).x;

	outFragColor = texture(velocityFieldTex, inUV);
	// Subtract the pressure gradient from velocity
	outFragColor.xy -= (ubo.timestep / (2.f * dudv)) * vec2(pR - pL, pT - pB);
}
