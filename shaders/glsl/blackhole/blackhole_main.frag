#version 450

// out
layout (location = 0) out vec4 outFragColor;

// layout (binding = 1) uniform samplerCube samplerCubeMap;

void main() {
	outFragColor = vec4(vec3(0.f, 0.f, 1.f), 1.f);
}
