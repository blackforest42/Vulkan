#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inUVW;

// out
layout (location = 0) out vec3 outPos;
layout (location = 1) out vec3 outUVW;

layout (binding = 0) uniform RayMarchUBO
{
	mat4 model;
	mat4 invModel;
	mat4 cameraView;
	mat4 perspective;
    vec3 cameraPos;
    vec2 screenRes;
    float time;
    int toggleView;
} ubo;


void main() {
    outPos = vec3(ubo.model * vec4(inPos, 1.f));
    outUVW = inUVW;
    gl_Position = ubo.perspective * ubo.cameraView * ubo.model * vec4(inPos, 1.);
}

