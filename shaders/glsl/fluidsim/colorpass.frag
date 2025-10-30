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
	float norm = length(vector);
	vector = normalize(vector);
	vec2 result = (vector + vec2(1.0)) / 2.f;
	return result * norm;
}

void main() {
	vec3 texel = texture(velocityFieldTex, inUV).rgb;
	// if (texel.x < 0 && texel.y < 0) {
	//	debugPrintfEXT("Negative vector: %1.2v3f", texel);
	// }

	vec2 normalized = vector2color(texel.xy);
	float blue = (abs(normalized.x) + abs(normalized.y)) / 2.f;

	// drop 'blue' channel from result
	outFragColor = vec4(normalized, blue, 1.f);
}