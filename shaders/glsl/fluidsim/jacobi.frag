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

layout (binding = 1) uniform sampler2D pressureTex;
layout (binding = 2) uniform sampler2D divergenceTex;


void main() {
	// For whatever reason, the duvdv has to be multiplied by 2.f
	// otherwise the velocity field turns from smooth -> noise/chaos
	vec2 dudv = 2.f / ubo.bufferResolution;
	float du = dudv.x;
	float dv = dudv.y;

	// Pressure
	vec4 xL = texture(pressureTex, (inUV - vec2(du, 0)));
	vec4 xR = texture(pressureTex, (inUV + vec2(du, 0)));
	vec4 xB = texture(pressureTex, (inUV - vec2(0, dv)));
	vec4 xT = texture(pressureTex, (inUV + vec2(0, dv)));

	// Divergence
	vec4 bC = texture(divergenceTex, inUV);

	// evaluate Jacobi iteration
	outFragColor = (xL + xR + xB + xT + bC) * 0.25f;

}
