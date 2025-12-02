#version 450

// in
layout (location = 0) in vec2 inUV[];
 
// out
layout (vertices = 4) out;
layout (location = 0) out vec2 outUV[4];

layout(set = 0, binding = 0) uniform UBO
{
	mat4 mvp;
} ubo;

layout(set = 0, binding = 1) uniform sampler2D heightMap;

void main(void)
{
	if (gl_InvocationID == 0)
	{

		// Tessellation factor can be set to zero by example
		// to demonstrate a simple passthrough
		gl_TessLevelInner[0] = 16.0;
		gl_TessLevelInner[1] = 16.0;
		gl_TessLevelOuter[0] = 16.0;
		gl_TessLevelOuter[1] = 16.0;
		gl_TessLevelOuter[2] = 16.0;
		gl_TessLevelOuter[3] = 16.0;
	}

	gl_out[gl_InvocationID].gl_Position =  gl_in[gl_InvocationID].gl_Position;
	outUV[gl_InvocationID] = inUV[gl_InvocationID];

}