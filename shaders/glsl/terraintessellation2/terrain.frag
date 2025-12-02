#version 450

// in
layout (location = 1) out vec2 inUV;

//out

layout (set = 0, binding = 1) uniform sampler2D samplerHeight; 

layout (location = 0) out vec4 outFragColor;

void main(void)
{
	// Get height from displacement map
	float h = textureLod(samplerHeight, inUV, 0.0).r;
	outFragColor = vec4(h, h, h, 1);
}