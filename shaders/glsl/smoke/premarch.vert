#version 450

layout(location = 0) in vec3 inPosition;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 worldViewProjection;
    mat4 invWorldViewProjection;
    vec3 eyePosition;
    vec3 volumeMax;
    vec3 volumeMin;
    vec2 screenRes;
    float nearPlane;
    float farPlane;
    float opacityModulator;
    float gridScaleFactor;
    int renderBackFaces;  // 1 = back faces, 0 = front faces
} ubo;

layout(location = 0) out vec3 outTexCoords;
layout(location = 1) out float outDepth;

void main() {
    // Transform vertex to world space (scale from [0,1] to volume bounds)
    vec3 worldPos = mix(ubo.volumeMin, ubo.volumeMax, inPosition);
    vec4 clipPos = ubo.worldViewProjection * vec4(worldPos, 1.0);
    
    gl_Position = clipPos;
    
    // Always output texture coords and depth
    // Fragment shader will decide what to do with them
    outTexCoords = inPosition;  // [0,1] texture coordinates
    outDepth = clipPos.w;       // View-space depth
}