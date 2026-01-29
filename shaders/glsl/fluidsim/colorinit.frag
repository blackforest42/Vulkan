#version 450
#extension GL_EXT_debug_printf : enable

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBO
{
    int whichTexture;
} ubo;

const float PI = 3.14159265358979323846;

void main() {
    float x = inUV.x;
    float y = inUV.y;

    // init color map as checkerboard
    float r, g, b;
    r = step(1.0, mod(floor((x + 1.0) / 0.2) + floor((y + 1.0) / 0.2), 2.0));
    b = step(1.0, mod(floor((x + 1.0) / 0.3) + floor((y + 1.0) / 0.3), 2.0));
    g = step(1.0, mod(floor((x + 1.0) / 0.4) + floor((y + 1.0) / 0.4), 2.0));
    outFragColor.rgb = vec3(r, g, b);
    return;
}