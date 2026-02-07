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
    float pixels_per_edge;
} ubo;

layout(binding = 2) uniform WaveParams {
    vec4 frequency[4];
    vec4 amplitude[4];
    vec4 directionX[4];
    vec4 directionY[4];
    vec4 phase[4];

    float time;
    float chopiness;
    float noiseStrength;
    float rippleScale;

    vec2 windDirection;
    float angleDeviation;
    float speedDeviation;

    float gravity;
    float minWavelength;
    float maxWavelength;
    float amplitudeOverLength;

    float _padding[2];
} waves;

layout(binding = 5) uniform sampler2D heightMap;
layout(binding = 6) uniform sampler2D slopeMap;

void main() {
    // --- 1. Bilinear Interpolation of Patch Corners ---
    // Interpolate texture coordinates from the 4 patch control points
    vec2 uv1 = mix(inUV[0], inUV[1], gl_TessCoord.x);
    vec2 uv2 = mix(inUV[3], inUV[2], gl_TessCoord.x);
    outUV = mix(uv1, uv2, gl_TessCoord.y);

    // --- 2. Calculate Base World Position ---
    // Scale texture coordinates to world space
    // Assuming the base mesh spans [-worldSize/2, worldSize/2] on XZ plane
    const float worldSize = 100.0;
    vec2 xzPos = outUV * worldSize - worldSize * 0.5;
    vec3 basePos = vec3(xzPos.x, 0.0, xzPos.y);// X and Z horizontal, Y=0

    // --- 3. Apply FFT Heightfield Displacement ---
    float heightSample = texture(heightMap, outUV).r;
    vec3 displacedPos = basePos + vec3(0.0, heightSample, 0.0);

    // --- 4. Transform to World Space ---
    outWorldCoord = displacedPos;

    // --- 5. Build Tangent-Bitangent-Normal Matrix ---
    vec2 slope = texture(slopeMap, outUV).rg;
    vec3 tangent = normalize(vec3(1.0, slope.x, 0.0));
    vec3 bitangent = normalize(vec3(0.0, slope.y, 1.0));
    vec3 geometricNormal = normalize(cross(tangent, bitangent));

    // Normalize tangent space basis vectors
    tangent = normalize(tangent);
    bitangent = normalize(bitangent);

    // Construct TBN matrix for transforming normal map to world space
    outTBN = mat3(tangent, bitangent, geometricNormal);

    // --- 6. Calculate Final Screen Position ---
    gl_Position = ubo.perspective * ubo.view * vec4(outWorldCoord, 1.0);
}
