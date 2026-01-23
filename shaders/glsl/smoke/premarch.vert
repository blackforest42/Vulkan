#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inUVW;

layout(location = 0) out vec3 outUVW;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 mvp;
    mat4 invMvp;
    vec3 cameraPosition;
} ubo;

void main() {
    vec4 clipPos = ubo.mvp * vec4(inPosition, 1.0);
    gl_Position = clipPos;

    outUVW = inUVW;
}