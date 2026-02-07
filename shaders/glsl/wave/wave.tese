#version 450

// in
layout (location = 0) in vec2 inUV[];

// out
layout (location = 0) out vec2 outUV;
layout (location = 1) out vec3 outWorldCoord;
layout (location = 2) out mat3 outTBN;

layout(quads, fractional_odd_spacing, ccw) in;

layout(set = 0, binding = 0) uniform UBO
{
    mat4 perspective;
    mat4 view;
    vec3 camera_pos;
    vec2 screen_res;
    float grid_scale;
} ubo;

layout(binding = 2) uniform OceanParams {
    vec4 wind_dir_speed_amp;
    vec4 time_patch_chop_height;
    ivec4 grid;
} ocean;

layout(binding = 3) uniform sampler2D heightMap;

void main() {
    // --- 1. Bilinear Interpolation of Patch Corners ---
    // Interpolate texture coordinates from the 4 patch control points
    vec2 uv1 = mix(inUV[0], inUV[1], gl_TessCoord.x);
    vec2 uv2 = mix(inUV[3], inUV[2], gl_TessCoord.x);
    outUV = mix(uv1, uv2, gl_TessCoord.y);

    // --- 2. Calculate Base World Position ---
    vec2 xzPos = outUV * ubo.grid_scale - ubo.grid_scale * 0.5;

    float height_scale = ocean.time_patch_chop_height.w;
    float h = texture(heightMap, outUV).r * height_scale;

    vec3 displacedPos = vec3(xzPos.x, h, xzPos.y);
    outWorldCoord = displacedPos;

    // --- 3. Build Tangent-Bitangent-Normal Matrix from height map ---
    float du = 1.0 / float(ocean.grid.x);
    vec2 uvR = outUV + vec2(du, 0.0);
    vec2 uvL = outUV - vec2(du, 0.0);
    vec2 uvU = outUV + vec2(0.0, du);
    vec2 uvD = outUV - vec2(0.0, du);

    float hR = texture(heightMap, uvR).r * height_scale;
    float hL = texture(heightMap, uvL).r * height_scale;
    float hU = texture(heightMap, uvU).r * height_scale;
    float hD = texture(heightMap, uvD).r * height_scale;

    float dx = 2.0 * du * ubo.grid_scale;
    float dz = 2.0 * du * ubo.grid_scale;

    vec3 tangent = normalize(vec3(dx, hR - hL, 0.0));
    vec3 bitangent = normalize(vec3(0.0, hU - hD, dz));
    vec3 geometricNormal = normalize(cross(tangent, bitangent));

    outTBN = mat3(tangent, bitangent, geometricNormal);

    // --- 4. Calculate Final Screen Position ---
    gl_Position = ubo.perspective * ubo.view * vec4(outWorldCoord, 1.0);
}
