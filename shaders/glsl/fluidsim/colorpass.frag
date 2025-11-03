#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;


layout (binding = 0) uniform UBO
{
	int chooseTextureMap;
} ubo;

layout (binding = 1) uniform sampler2D colorFieldTex;
layout (binding = 2) uniform sampler2D velocityFieldTex;
layout (binding = 3) uniform sampler2D pressureFieldTex;

const float PI = 3.14159265358979323846;

// takes an un-normalized vec2 and converts to normalized [0 - 1] vec3
vec3 vector2color(vec2 vector) {
	float magnitude = length(vector);

	float rad = atan(vector.y, vector.x);
	rad += PI;
	float deg = rad * 180 / PI;
	float u = 0, v = 0, w = 0;
	if (0 <= deg && deg < 120) {
		u = deg / 120.f;
		v = 1 - u;
	}
	else if (120 <= deg && deg < 240) {
		deg -= 120;
		w = deg / 120.f;
		u = 1 - w;
	}
	else {
		deg -= 240;
		v = deg / 120.f;
		w = 1 - v;
	}

	vec3 result = vec3(1.0 * u, 1.0f * v, 1.f * w);

	return result * magnitude;
}

void main() {
	vec3 result;
	vec4 texel;
	switch (ubo.chooseTextureMap) {
		case 0:
			result = texture(colorFieldTex, inUV).rgb;
			break;
		case 1:
			texel = texture(velocityFieldTex, inUV);
			result = vector2color(texel.xy);
			break;
		default: {
			texel = texture(pressureFieldTex, inUV);
			result = vector2color(texel.xy);
			break;
		}
	}

	// drop 'blue' channel from result
	outFragColor = vec4(result, 1.f);
}