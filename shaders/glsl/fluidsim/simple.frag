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
    vec2 uv = gl_FragCoord.xy / ubo.viewportResolution.xy;

	vec3 green = vec3(0.0, 1., 0.);
	vec3 blue = vec3(0., 0., 1.);
	if (is_near(uv.x, 0) || is_near(uv.x, 1) || is_near(uv.y, 0) || is_near(uv.y, 1)) {
		// debugPrintfEXT("My vector is %v2f", uv);
		outFragColor = vec4(green, 1.0);
	} else {
		outFragColor = vec4(blue, 1.0);
	}

}
