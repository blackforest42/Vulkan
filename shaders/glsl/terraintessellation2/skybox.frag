#version 450

// in
layout (location = 0) in vec3 inUVW;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 1) uniform samplerCube samplerCubeMap;

void main() 
{
	outFragColor = texture(samplerCubeMap, inUVW);
}