#version 450

// in
layout (location = 1) in vec2 inUV[];
 
// out
layout (location = 1) out vec2 outUV;

layout (set = 0, binding = 0) uniform UBO 
{
	mat4 mvp;
} ubo; 

layout(set = 0, binding = 1) uniform sampler2D heightMap;

layout(quads, equal_spacing, cw) in;

void main()
{
	// Interpolate UV coordinates
	vec2 uv1 = mix(inUV[0], inUV[1], gl_TessCoord.x);
	vec2 uv2 = mix(inUV[3], inUV[2], gl_TessCoord.x);
	outUV = mix(uv1, uv2, gl_TessCoord.y);

	// Interpolate positions
	vec4 pos1 = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);
	vec4 pos2 = mix(gl_in[3].gl_Position, gl_in[2].gl_Position, gl_TessCoord.x);
	vec4 pos = mix(pos1, pos2, gl_TessCoord.y);
	// Displace
	pos.y -= textureLod(heightMap, outUV, 0.0).x;

	// Perspective projection
	gl_Position = ubo.mvp * pos;

}