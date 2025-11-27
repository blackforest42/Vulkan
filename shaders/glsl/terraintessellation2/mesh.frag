#version 450

layout (location = 0) in float height;

layout (location = 0) out vec4 outFragColor;

void main(void)
{
	// shift and scale the height into a grayscale value
    float h = (-height + 16)/32.0f;
	outFragColor = vec4(h, h, h, 1);
}