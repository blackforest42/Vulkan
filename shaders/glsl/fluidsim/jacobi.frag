#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBO
{
	float alpha;
	float rBeta;
} ubo;

layout (binding = 1) uniform sampler2D field1; // 'x' texture
layout (binding = 2) uniform sampler2D field2; // 'b' texture


void main() {
	/* debug
	vec3 black = vec3(0.0);
	vec3 red = vec3(1.0, 0., 0.);
	vec3 green = vec3(0.0, 1., 0.);
	vec3 blue = vec3(0., 0., 1.);
	outFragColor = vec4(black, 1.0);
	*/

	// left, right, bottom, and top x samples
	vec4 xL = texture(field1, inUV - vec2(1, 0));
	vec4 xR = texture(field1, inUV + vec2(1, 0));
	vec4 xB = texture(field1, inUV - vec2(0, 1));
	vec4 xT = texture(field1, inUV + vec2(0, 1));
	// b sample, from center
	vec4 bC = texture(field2, inUV);
	// evaluate Jacobi iteration
	outFragColor = (xL + xR + xB + xT + ubo.alpha * bC) * ubo.rBeta;

}
