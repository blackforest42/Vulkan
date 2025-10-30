#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBO
{
	vec3 color;
    vec2 epicenter;
	vec2 viewportResolution;
	float radius;
} ubo;

layout (binding = 1) uniform sampler2D velocityField;

float gaussian(vec2 pos, float radius) {
	return exp(-dot(pos, pos) / radius);
}

void main() {
	vec2 delta_distance = inUV - ubo.epicenter / ubo.viewportResolution;
	float rad = ubo.viewportResolution.x * ubo.radius;
	outFragColor = texture(velocityField, inUV) + ubo.color.xyzz * gaussian(delta_distance, rad);
}
