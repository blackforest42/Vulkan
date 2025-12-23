#version 450
#extension GL_EXT_debug_printf : enable

layout (location = 0) in vec3 inPos;

layout (binding = 0) uniform UBOView
{
	mat4 projection;
	mat4 view;
    mat4 invModelView;
    vec3 cameraPos;
} uboView;


layout (binding = 1) uniform UBOModel
{
	mat4 model; 
} uboModel;

layout (location = 0) out vec3 lookAt;

void main() 
{
	gl_Position = uboView.projection * uboView.view * uboModel.model * vec4(inPos, 1.0);
	vec3 translation = uboModel.model[3].xyz;
	vec3 voxel_position = inPos + translation;
	// Create a direction vector from camera to fragment
	// vec3(0) is the camera's position in view space
	lookAt = voxel_position - (uboView.invModelView * vec4(vec3(0), 1)).xyz;
}
