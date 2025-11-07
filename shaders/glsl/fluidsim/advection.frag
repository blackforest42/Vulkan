#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBO
{
	vec2 bufferResolution;
	float timestep;
} ubo;

layout (binding = 1) uniform sampler2D velocityTex;
layout (binding = 2) uniform sampler2D textureToAdvect;

void main() {
	float x = inUV.x;
	float y = inUV.y;

	// follow the velocity field "back in time"
	vec2 pos = (inUV - 0.5 * ubo.timestep * texture(velocityTex, inUV).xy);
	// interpolate and write to the output fragment
	outFragColor = texture(textureToAdvect, pos);

}
