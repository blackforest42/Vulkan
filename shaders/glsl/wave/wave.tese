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

const float PI = 3.14159265359;
const int NUM_GEO_WAVES = 4;// Use first 4 waves for geometry

vec3 gerstnerWave(vec2 pos, out vec3 tangent, out vec3 bitangent);
float calcDepthFilter(float depth, float depthScale);

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

    // --- 3. Apply Gerstner Wave Displacement ---
    vec3 tangent, bitangent;
    vec3 waveDisplacement = gerstnerWave(vec2(basePos.x, basePos.z), tangent, bitangent);

    vec3 displacedPos = basePos + waveDisplacement;

    // --- 4. Transform to World Space ---
    outWorldCoord = displacedPos;

    // --- 5. Build Tangent-Bitangent-Normal Matrix ---
    // The geometric normal from Gerstner waves
    vec3 geometricNormal = normalize(cross(tangent, bitangent));

    // Normalize tangent space basis vectors
    tangent = normalize(tangent);
    bitangent = normalize(bitangent);

    // Construct TBN matrix for transforming normal map to world space
    outTBN = mat3(tangent, bitangent, geometricNormal);

    // --- 6. Calculate Final Screen Position ---
    gl_Position = ubo.perspective * ubo.view * vec4(outWorldCoord, 1.0);
}

// Calculate Gerstner wave displacement and derivatives
// Returns: displacement vector and modifies tangent/bitangent for TBN
// pos: 2D horizontal position (X, Z)
// Returns: displacement in (dx, dy, dz) where Y is vertical
vec3 gerstnerWave(vec2 pos, out vec3 tangent, out vec3 bitangent) {
    vec3 displacement = vec3(0.0);
    tangent = vec3(1.0, 0.0, 0.0);// Initially along X axis
    bitangent = vec3(0.0, 0.0, 1.0);// Initially along Z axis

    // Calculate only the first 4 waves for geometric displacement
    // (The remaining 12 are handled by the normal map in the fragment shader)
    for (int i = 0; i < NUM_GEO_WAVES; i++) {
        int vecIdx = i / 4;// Always 0 for first 4 waves
        int comp = i % 4;

        // Extract wave parameters
        float freq = waves.frequency[vecIdx][comp];
        float amp = waves.amplitude[vecIdx][comp];
        vec2 dir = vec2(waves.directionX[vecIdx][comp],
        waves.directionY[vecIdx][comp]);
        float ph = waves.phase[vecIdx][comp];

        // Normalize direction vector (this is horizontal direction in XZ plane)
        dir = normalize(dir);

        // Calculate wave phase: k·position + ωt
        // pos is (X, Z) horizontal coordinates
        float wavePhase = dot(dir, pos) * freq + ph * 2.0 * PI;

        float sinWave = sin(wavePhase);
        float cosWave = cos(wavePhase);

        // --- Vertical Displacement (Height) ---
        // Y is up, so we modify the Y component
        displacement.y += amp * sinWave;

        // --- Horizontal Displacement (Chop/Steepness) ---
        // This creates the sharp wave peaks characteristic of Gerstner waves
        // K factor limits maximum chop to prevent self-intersection
        float K = waves.chopiness;

        // Limit K to prevent folding: K_max = 1/(2π * amp_sum)
        if (waves.amplitudeOverLength > 0.0) {
            float maxK = waves.chopiness /
            (2.0 * PI * waves.amplitudeOverLength * float(NUM_GEO_WAVES));
            K = min(K, maxK);
        }

        // Horizontal displacement in XZ plane
        // dir.x affects X, dir.y affects Z
        displacement.x += dir.x * K * amp * cosWave;
        displacement.z += dir.y * K * amp * cosWave;

        // --- Tangent Space Derivatives ---
        // These are needed to compute the geometric normal

        // Precompute common terms
        float wa = freq * amp;// ω * A
        float kwa = K * wa;// K * ω * A

        // Tangent (derivative with respect to X)
        // ∂P/∂x in (X, Y, Z) coordinates
        tangent.x -= dir.x * dir.x * kwa * sinWave;// X component
        tangent.y += dir.x * wa * cosWave;// Y component (vertical)
        tangent.z -= dir.x * dir.y * kwa * sinWave;// Z component

        // Bitangent (derivative with respect to Z)
        // ∂P/∂z in (X, Y, Z) coordinates
        bitangent.x -= dir.x * dir.y * kwa * sinWave;// X component
        bitangent.y += dir.y * wa * cosWave;// Y component (vertical)
        bitangent.z -= dir.y * dir.y * kwa * sinWave;// Z component
    }

    return displacement;
}
