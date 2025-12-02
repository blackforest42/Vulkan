#version 450

// in
layout (location = 0) in vec3 inPos;
layout (location = 1) in vec2 inUV;

// out
layout (location = 0) out vec2 outUV;


void main(void)
{
	outUV = inUV;
	gl_Position = vec4(inPos, 1.0);
}