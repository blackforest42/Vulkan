#version 450

// in
layout (location = 0) in vec2 inUV;
layout (location = 1) in vec3 inWorldCoord;
layout (location = 2) in mat3 inTBN;

// out
layout (location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform UBO
{
    mat4 perspective;
    mat4 view;
    vec3 camera_pos;
    vec2 screen_res;
    float grid_scale;
} ubo;

layout(binding = 3) uniform sampler2D normalMap;
layout(binding = 4) uniform samplerCube cubeMap;

const vec4 waterTint = vec4(0.0, 0.3, 0.4, 0.8);

void main() {
    // Sample high-frequency normals from compute shader
    vec3 bumpNormal = texture(normalMap, inUV * 25.0).xyz * 2.0 - 1.0;

    // Transform to world space
    vec3 normal = normalize(inTBN * bumpNormal);

    // Reflection calculation
    vec3 viewDir = normalize(inWorldCoord - ubo.camera_pos);
    vec3 reflectDir = reflect(viewDir, normal);

    // Sample environment map (swap Y and Z for D3D→Vulkan cubemap convention)
    vec3 envSample = texture(cubeMap, vec3(reflectDir.x, -reflectDir.z, reflectDir.y)).rgb;

    // Fresnel approximation
    float fresnel = pow(1.0 - max(dot(-viewDir, normal), 0.0), 5.0);

    // Reduce reflection strength for a less specular surface
    float reflectivity = mix(0.02, 0.35, fresnel);
    vec3 color = mix(waterTint.rgb, envSample * 0.85, reflectivity);

    // Depth-based alpha for shorelines (requires depth texture)
    float alpha = waterTint.a;

    fragColor = vec4(color, alpha);
}