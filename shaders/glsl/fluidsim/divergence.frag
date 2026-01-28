#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

layout (std140, binding = 0) uniform UBO
{
    vec2 bufferResolution;
    float timestep;
} ubo;

layout (binding = 1) uniform sampler2D vectorField;

void main() {
    vec2 dudv = 1. / ubo.bufferResolution;
    float du = dudv.x;
    float dv = dudv.y;

    vec2 x_negative = inUV - vec2(du, 0);
    vec2 x_positive = inUV + vec2(du, 0);
    vec2 y_positive = inUV - vec2(0, dv);
    vec2 y_negative = inUV + vec2(0, dv);

    vec4 wL = texture(vectorField, x_negative);
    vec4 wR = texture(vectorField, x_positive);
    vec4 wB = texture(vectorField, y_positive);
    vec4 wT = texture(vectorField, y_negative);

    float cx = (-2.0 * du / ubo.timestep);
    float cy = (-2.0 * dv / ubo.timestep);
    outFragColor = vec4((cx * (wR.x - wL.x) + cy * (wT.y - wB.y)));
}