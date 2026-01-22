#version 450

// ins
layout(location = 0) in vec3 inTexCoords;
layout(location = 1) in float inDepth;

// outs
layout(location = 0) out vec4 outRayData;

layout(binding = 0) uniform UniformBufferObject {
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
} ubo;

layout (push_constant) uniform PushConsts {
    int renderBackFaces;  // 1 = back faces, 0 = front faces
} pc;

void main() {
    // Sample scene depth
    //vec2 screenUV = gl_FragCoord.xy / ubo.screenRes;
    outRayData = vec4(inTexCoords, inDepth);
    return;
    
    if (pc.renderBackFaces == 1) {
        // ====================================================================
        // BACK FACE PASS
        // ====================================================================
    } else {
        // ====================================================================
        // FRONT FACE PASS
        // ====================================================================
    }
}