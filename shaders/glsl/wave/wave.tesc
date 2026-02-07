#version 450

// in
layout (location = 0) in vec2 inUV[];

// out
layout (vertices = 4) out;
layout (location = 0) out vec2 outUV[4];

layout(set = 0, binding = 0) uniform UBO
{
    mat4 perspective;
    mat4 view;
    vec3 camera_pos;
    vec2 screen_res;
    float grid_scale;
    vec3 patch_rotation;
} ubo;

// Tessellation configuration
layout(binding = 1) uniform TessellationConfig
{
    float min_tess_level;
    float max_tess_level;
    float min_distance;
    float max_distance;
    float frustum_cull_margin;
} tess_config;


const float EPSILON = 0.01;

void main(void)
{
    // Only first invocation calculates tessellation levels
    if (gl_InvocationID == 0)
    {
        // --- Calculate Tessellation Levels ---

        // Step 1: transform each vertex into camera space
        vec4 eyeSpacePos00 = ubo.view * gl_in[0].gl_Position;
        vec4 eyeSpacePos01 = ubo.view * gl_in[1].gl_Position;
        vec4 eyeSpacePos10 = ubo.view * gl_in[2].gl_Position;
        vec4 eyeSpacePos11 = ubo.view * gl_in[3].gl_Position;

        // Step 2: "distance" from camera scaled between 0 and 1
        float distance00 = clamp(
        (abs(eyeSpacePos00.z) - tess_config.min_distance) / (tess_config.max_distance - tess_config.min_distance),
        0.0, 1.0);
        float distance01 = clamp(
        (abs(eyeSpacePos01.z) - tess_config.min_distance) / (tess_config.max_distance - tess_config.min_distance),
        0.0, 1.0);
        float distance10 = clamp(
        (abs(eyeSpacePos10.z) - tess_config.min_distance) / (tess_config.max_distance - tess_config.min_distance),
        0.0, 1.0);
        float distance11 = clamp(
        (abs(eyeSpacePos11.z) - tess_config.min_distance) / (tess_config.max_distance - tess_config.min_distance),
        0.0, 1.0);

        // Step 3: interpolate edge tessellation level based on closer vertex
        float tessLevel0 = mix(tess_config.max_tess_level, tess_config.min_tess_level, min(distance10, distance00));
        float tessLevel1 = mix(tess_config.max_tess_level, tess_config.min_tess_level, min(distance00, distance01));
        float tessLevel2 = mix(tess_config.max_tess_level, tess_config.min_tess_level, min(distance01, distance11));
        float tessLevel3 = mix(tess_config.max_tess_level, tess_config.min_tess_level, min(distance11, distance10));

        // Step 4: Set tessellation levels
        gl_TessLevelOuter[0] = tessLevel0;
        gl_TessLevelOuter[1] = tessLevel1;
        gl_TessLevelOuter[2] = tessLevel2;
        gl_TessLevelOuter[3] = tessLevel3;

        // Step 5: set the inner tessellation levels to the max of the two parallel edges
        gl_TessLevelInner[0] = max(tessLevel1, tessLevel3);
        gl_TessLevelInner[1] = max(tessLevel0, tessLevel2);
    }

    // Pass through data for all invocations
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    outUV[gl_InvocationID] = inUV[gl_InvocationID];
}
