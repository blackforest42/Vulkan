#version 450

layout (location = 0) in vec3 inPos;

layout (set = 0, binding = 0) uniform SunUBO
{
    mat4 mvp;
    vec4 color;
} ubo;

layout (location = 0) out vec4 outColor;

void main()
{
    outColor = ubo.color;
    gl_Position = ubo.mvp * vec4(inPos, 1.0);
}
