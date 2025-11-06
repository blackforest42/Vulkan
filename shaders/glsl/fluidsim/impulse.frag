#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBO
{
    vec2 epicenter;
	vec2 bufferRes;
    vec2 dxdy;
	float radius;
} ubo;

layout (binding = 1) uniform sampler2D velocityField;

float gaussian(vec2 pos, float radius) {
	return exp(-dot(pos, pos) / radius);
}

void main() {
	// debug
	// debugPrintfEXT("impulse frag shader called");
	vec2 dxdy = ubo.dxdy / ubo.bufferRes;
	float dx = dxdy.x;
	float dy = dxdy.y;

	vec2 delta_distance = inUV - ubo.epicenter / ubo.bufferRes;

	float rad = ubo.radius;
	outFragColor = texture(velocityField, inUV) + 
		vec4(-10 * dxdy.x, -10 * dxdy.y, 0, 1) * gaussian(delta_distance, rad);
}
