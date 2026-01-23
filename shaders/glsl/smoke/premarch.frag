#version 450

// ins
layout(location = 0) in vec3 inUVW;

// outs
layout(location = 0) out vec4 outRayData;

layout(binding = 0) uniform UniformBufferObject {
    mat4 mvp;
    mat4 invMvp;
    vec3 cameraPosition;
} ubo;

layout (push_constant) uniform PushConsts {
    int renderBackFaces;// 1 = back faces, 0 = front faces
} pc;

void main() {
    outRayData = vec4(inUVW, 1.f);
    return;
}