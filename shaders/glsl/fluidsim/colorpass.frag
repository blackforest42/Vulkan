#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;


layout (binding = 0) uniform sampler2D velocityFieldTex;
layout (binding = 1) uniform sampler2D pressureFieldTex;

// takes an un-normalized vec2 and converts to normalized [0 - 1] vec3
vec2 vector2color(vec2 vector) {
	vector = normalize(vector);
	vec2 result = (vector + vec2(1.0)) / 2.f;
	return result;
}

void main() {
	vec4 texel = texture(velocityFieldTex, inUV);
	vec2 normalized_rgb = vector2color(texel.xy);
	outFragColor = vec4(0, 0, 0, 1);
	// Preference: drop the 'red' channel, keep green and blue
	outFragColor.rgb = vec3(0.0f, normalized_rgb);
}