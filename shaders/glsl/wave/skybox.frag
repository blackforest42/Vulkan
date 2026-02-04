#version 450

// in
layout (location = 0) in vec3 inUVW;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 1) uniform samplerCube samplerCubeMap;

void main() 
{
	// gamma correct
	vec3 fragColor = texture(samplerCubeMap, inUVW).xyz;
	fragColor = pow(fragColor, vec3(1.0 / 2.2f));
	outFragColor.rgb = fragColor;
}