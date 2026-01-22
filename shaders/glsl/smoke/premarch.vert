#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inUVW;

layout(location = 0) out vec3 outUVW;
layout(location = 1) out float outDepth;

layout(binding = 0) uniform UniformBufferObject {
    mat4 mvp;
    mat4 invMvp;
    vec3 cameraPosition;
    vec3 volumeMax;
    vec3 volumeMin;
} ubo;

void main() {
    // Transform vertex to world space (scale from [0,1] to volume bounds)
    vec3 worldPos = mix(ubo.volumeMin, ubo.volumeMax, inPosition);
    vec4 clipPos = ubo.mvp * vec4(worldPos, 1.0);
    gl_Position = clipPos;
    
    outUVW = inUVW;
    outDepth = clipPos.w;       // View-space depth
}