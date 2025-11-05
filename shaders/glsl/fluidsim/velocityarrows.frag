#version 450

// in
layout(location = 0) in vec2 inUV;

// out
layout(location = 0) out vec4 outFragColor;

void main() {
    // red triangle
    outFragColor = vec4(1.0f, 0, 0, 1.f);
}