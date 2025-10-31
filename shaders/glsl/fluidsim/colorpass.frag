#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;


layout (binding = 0) uniform sampler2D velocityFieldTex;
layout (binding = 1) uniform sampler2D pressureFieldTex;

const float PI = 3.14159;

// takes an un-normalized vec2 and converts to normalized [0 - 1] vec3
vec3 vector2color(vec2 vector) {
	float norm = length(vector);
	vec3 result = vec3(normalize(vector), 0);

	float blue = 0;
	if (vector.x < 0 && vector.y < 0) {
		blue = abs(vector.x + vector.y) / 2f;
	}
	result.b = blue;

	result = (result + vec3(1.0)) / 2.f;

	return result * norm;
}

void main() {
	vec3 texel = texture(velocityFieldTex, inUV).rgb;
	// if (texel.x < 0 && texel.y < 0) {
	//	debugPrintfEXT("Negative vector: %1.2v3f", texel);
	// }

	vec3 normalized = vector2color(texel.xy);

	// drop 'blue' channel from result
	outFragColor = vec4(normalized, 1.f);
}