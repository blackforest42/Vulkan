#version 450

// in
layout(location = 0) in vec2 inPosition;

// out

void main() {
    gl_Position = vec4(inPosition, 0.0, 1.0);
}