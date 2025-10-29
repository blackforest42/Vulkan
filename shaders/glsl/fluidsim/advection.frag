#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBO
{
    vec2 viewportResolution;
	float timestep;
} ubo;

layout (binding = 1) uniform sampler2D field1;
layout (binding = 2) uniform sampler2D field2;

void main() {
	/* debug */
	vec3 black = vec3(0.0);
	vec3 red = vec3(1.0, 0., 0.);
	vec3 green = vec3(0.0, 1., 0.);
	vec3 blue = vec3(0., 0., 1.);
	float x = gl_FragCoord.x - 0.5;
	float y = gl_FragCoord.y - 0.5;

	vec2 coords = gl_FragCoord.xy;

	// boundary
	if (x == 0 || x == ubo.viewportResolution.x - 1 || y == 0 || y == ubo.viewportResolution.y - 1) {
		//debugPrintfEXT("My value is: %f and %f", x, y);
		outFragColor = vec4(blue, 1.0);
		return; 
	}
	//outFragColor = vec4(green, 1.f);
	// return;

	// follow the velocity field "back in time"
	vec2 pos = coords - ubo.timestep * texture(field1, coords).xy;
	// interpolate and write to the output fragment
	outFragColor = texture(field2, pos);

}
