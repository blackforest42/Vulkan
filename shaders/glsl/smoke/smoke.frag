#version 450

layout (location = 0) out vec4 outFragColor;

layout (location = 0) in vec3 lookAt;

void main() 
{
	outFragColor = vec4(1.0, 0, 0, .3f);
}