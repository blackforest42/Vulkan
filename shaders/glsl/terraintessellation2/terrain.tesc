#version 450

// in
layout (location = 1) in vec2 inUV[];
 
// out
layout (vertices = 4) out;
layout (location = 1) out vec2 outUV[4];

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
		gl_TessLevelInner[0] = 1.0;
		gl_TessLevelInner[1] = 1.0;
		gl_TessLevelOuter[0] = 1.0;
		gl_TessLevelOuter[1] = 1.0;
		gl_TessLevelOuter[2] = 1.0;
		gl_TessLevelOuter[3] = 1.0;
	}

	gl_out[gl_InvocationID].gl_Position =  gl_in[gl_InvocationID].gl_Position;
	outUV[gl_InvocationID] = inUV[gl_InvocationID];

}