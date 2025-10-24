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



void main() {
	//debugPrintfEXT("My vector is %v2f", ubo.viewportResolution);
    vec2 uv = gl_FragCoord.xy / ubo.viewportResolution.xy - vec2(0.5);
    // Aspect ratio correction for non-square screens
	uv.x *= ubo.viewportResolution.x / ubo.viewportResolution.y;

	vec3 red = vec3(1.0, 0., 0.);
	vec3 blue = vec3(0., 0., 1.);
	outFragColor = vec4(blue, 1.0);
}
