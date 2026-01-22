#version 450

// ins
layout(location = 0) in vec3 inUVW;
layout(location = 1) in float inDepth;

// outs
layout(location = 0) out vec4 outRayData;

layout(binding = 0) uniform UniformBufferObject {
    mat4 mvp;
    mat4 invMvp;
    vec3 cameraPosition;
    vec3 volumeMax;
    vec3 volumeMin;
} ubo;

layout (push_constant) uniform PushConsts {
    int renderBackFaces;  // 1 = back faces, 0 = front faces
} pc;

void main() {
    // Sample scene depth
    //vec2 screenUV = gl_FragCoord.xy / ubo.screenRes;
    outRayData = vec4(inUVW, inDepth);
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