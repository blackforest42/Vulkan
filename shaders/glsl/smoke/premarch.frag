#version 450

//layout (location = 0) in vec2 outUV;

layout (location = 0) out vec4 outFragColor;


layout (binding = 0) uniform UBO
{
    vec2 screenRes;
} ubo;


void main() {
	outFragColor = vec4(1.0);
}

