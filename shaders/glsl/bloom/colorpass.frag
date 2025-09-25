#version 450

layout (binding = 1) uniform sampler2D colorMap;

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

bool tonemap = true;

void main() 
{
	if (tonemap) {
		float exposure = .1f;
		vec3 fragColor = inColor;
		fragColor = vec3(1.0) - exp(-fragColor * exposure);

		// gamma correct
		float gamma = 2.2f;
		fragColor = pow(fragColor, vec3(1.0 / gamma));

		outFragColor.rgb = fragColor;
	} else {
		outFragColor = vec4(inColor, 1.0);
		// outFragColor = texture(colorMap, inUV);// * vec4(inColor, 1.0);
	}
}