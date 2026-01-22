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
    int renderBackFaces;  // 1 = back faces, 0 = front faces
} ubo;

void main() {
    // Sample scene depth
    vec2 screenUV = gl_FragCoord.xy / ubo.screenRes;
    
    if (ubo.renderBackFaces == 1) {
        // ====================================================================
        // BACK FACE PASS
        // ====================================================================
        // Mark pixels where back faces were rendered (green channel negative)
        // xyz will be set by front faces, w = min(backface depth, scene depth)
        outRayData = vec4(0.0, -1.0, 0.0, inDepth);
    } else {
        // ====================================================================
        // FRONT FACE PASS
        // ====================================================================
        
        // Output negated texture coords (for subtractive blending)
        // and depth
        outRayData = vec4(-inTexCoords, inDepth);
    }
}