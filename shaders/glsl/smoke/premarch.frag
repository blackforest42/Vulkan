#version 450
#extension GL_EXT_debug_printf : enable

layout (location = 0) in vec3 inColor;

layout (location = 0) out vec4 outFragColor;

layout(push_constant) uniform PushConstants {
	int enableFrontMarch;
} pc;

layout (binding = 0) uniform UBO
{
    vec2 screenRes;
} ubo;


void main() {
	if (pc.enableFrontMarch == 1) {
		outFragColor.rgb = vec3(1.0);
	}
	else {
		outFragColor.rgb = inColor;
	}


}
