#version 450

layout (location = 0) in vec3 inPos;

layout (set = 0, binding = 0) uniform UBO 
{
	mat4 mvp;
} ubo;

layout (location = 0) out float height;

void main(void)
{
	height = inPos.y;
	gl_Position = ubo.mvp * vec4(inPos, 1.0);
}