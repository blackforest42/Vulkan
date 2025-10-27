#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBO
{
    vec2 viewportResolution;
} ubo;

const float EPS = 0.0005;

bool is_near(float x, float near) {
	return near - EPS <= x && x <= near + EPS;
}


void main() {
	float x = gl_FragCoord.x - 0.5;
	float y = gl_FragCoord.y - 0.5;

	vec3 green = vec3(0.0, 1., 0.);
	vec3 blue = vec3(0., 0., 1.);
	if (x == 0 || x == ubo.viewportResolution.x - 1 || y == 0 || y == ubo.viewportResolution.y - 1) {
		outFragColor = vec4(green, 1.0);
	} else {
		outFragColor = vec4(vec3(0), 1.0);
	}

}
