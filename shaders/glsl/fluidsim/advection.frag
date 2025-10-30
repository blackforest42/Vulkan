#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBO
{
	vec3 randomVec3;
    vec2 viewportResolution;
	float timestep;
} ubo;

layout (binding = 1) uniform sampler2D field1;
layout (binding = 2) uniform sampler2D field2;

// generates a rand number [0, 1]
float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
	// debug
	// outFragColor.xyz = ((vec3(rand(), rand(), rand())) * 100) - 50;
	// return;

	// follow the velocity field "back in time"
	vec2 pos = inUV - ubo.timestep * texture(field1, inUV).xy;
	// interpolate and write to the output fragment
	outFragColor = texture(field2, pos);

}
