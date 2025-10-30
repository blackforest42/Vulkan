#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform sampler2D velocityFieldTex;
layout (binding = 1) uniform sampler2D pressureFieldTex;

void main() {
	// Take the gradient of the scalar valued pressure field
	// Note: we only take the 'x' component since pressure field
	// is scalar valued and uniform.
	float pL = texture(pressureFieldTex, inUV - vec2(1, 0)).x;
	float pR = texture(pressureFieldTex, inUV + vec2(1, 0)).x;
	float pB = texture(pressureFieldTex, inUV - vec2(0, 1)).x;
	float pT = texture(pressureFieldTex, inUV + vec2(0, 1)).x;

	// Set output to the velocity field
	outFragColor = texture(velocityFieldTex, inUV);
	// Subtract the pressure gradient from velocity
	outFragColor.xy -= 0.5f * vec2(pR - pL, pT - pB);
}
