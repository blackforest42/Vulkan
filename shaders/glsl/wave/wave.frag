#version 450

// in
layout (location = 0) in vec2 inUV;

//out
layout (location = 0) out vec4 outFragColor;

void main(void)
{
    outFragColor = vec4(0, 0, 0, 1);
}