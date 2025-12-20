#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inColor;
layout (location = 3) in vec3 inNormal;

layout (binding = 0) uniform UBOView
{
	mat4 projection;
	mat4 view;
} uboView;


layout (binding = 1) uniform UBOModel
{
	mat4 model; 
} uboModel;

void main() 
{
	gl_Position = uboView.projection * uboView.view * uboModel.model * vec4(inPos, 1.0);
}
