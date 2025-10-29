#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;


layout (binding = 0) uniform sampler2D velocityFieldTex;
layout (binding = 1) uniform sampler2D pressureFieldTex;

// takes an un-normalized vec3 and converts to normalized [0 - 1] rgb3
vec3 vector2color(vec3 vector) {
	vector = normalize(vector);
	vec3 rgb = (vector + vec3(1.0)) / 2.f;
	return rgb;
}


void main() {
	vec2 coords = gl_FragCoord.xy;
	outFragColor.rgb = vector2color(texture(velocityFieldTex, inUV).xyz);
}