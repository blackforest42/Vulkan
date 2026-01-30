#version 450
#extension GL_EXT_nonuniform_qualifier : enable

// in
layout(location = 0) in vec2 inUV;

// out
layout(location = 0) out vec4 outFragColor;

// Texture index mappings
// 0 velocity
// 1 pressure
// 2 divergence
// 3 vorticity
// 4 density
// 5 temperature
layout(binding = 1) uniform sampler2D preMarchFrontTex;
layout(binding = 2) uniform sampler2D preMarchBackTex;
layout(binding = 3) uniform sampler3D readOnlyTexs[];

layout(binding = 0) uniform RayMarchUBO {
    mat4 cameraView;
    mat4 perspective;
    vec3 cameraPos;
    vec2 screenRes;
    float time;
    float rayStepSize;
    int toggleView;
    uint texId;
}
ubo;

float MAX_STEPS = 1 / ubo.rayStepSize;
const float SMOKE_DENSITY = 1.0f;
const float CLOUD_SIZE = .5f;
const vec3 SMOKE_COLOR = vec3(1.0);
const vec3 PRESSURE_COLOR = vec3(1.0, 0, 0);
const float INTENSITY_SCALE = 5.f;
const vec3 SUN_POSITION = vec3(1.0, 0.0, 0.0);

vec4 permute(vec4 x);
vec4 taylorInvSqrt(vec4 r);
float snoise(vec3 pos);
float fbm(vec3 pos);
vec4 rayMarchScalar(vec3 rayOrigin, vec3 rayDirection);
vec4 rayMarchSDF(vec3 rayOrigin, vec3 rayDir);
vec4 rayMarchVelocity(vec3 rayOrigin, vec3 rayDir);
bool intersectBox(vec3 rayOrigin, vec3 rayDir, out float tNear, out float tFar);
vec3 getBlackbodyColor(float t);

void main() {
    vec4 cloud = vec4(0);
    vec3 cubeRayEntry = texture(preMarchFrontTex, inUV).xyz;
    vec3 rayDir;
    if (cubeRayEntry == vec3(0)) {
        // ray misses the cube
        vec3 exit = vec3(inUV, 1);
        vec3 entry = vec3(0.5, 0.5, 0);
        rayDir = normalize(vec3(exit - entry));
    } else {
        // ray hits the cube
        vec3 cubeRayExit = texture(preMarchBackTex, inUV).xyz;
        rayDir = normalize(vec3(cubeRayExit - cubeRayEntry));
        if (ubo.toggleView == 0) {
            // March a texture
            if (ubo.texId == 0) {
                // March velocity smoke texture
                cloud = rayMarchVelocity(cubeRayEntry, rayDir);
            }
            else {
                // March scalar texture (e.g. smoke)
                cloud = rayMarchScalar(cubeRayEntry, rayDir);
            }
        } else if (ubo.toggleView == 1) {
            // March Noise
            cloud = rayMarchSDF(cubeRayEntry, rayDir);
        }
        else if (ubo.toggleView == 2) {
            cloud = vec4(cubeRayEntry, 1);
        }
        else if (ubo.toggleView == 3) {
            cloud = vec4(cubeRayExit, 1);
        }
    }

    // Sun and Sky
    vec3 sunDirection = normalize(SUN_POSITION);
    float sun = clamp(dot(sunDirection, rayDir), 0.0, 1.0);
    // Base sky color
    vec3 skyColor = vec3(0.7, 0.7, 0.90);
    // Add vertical gradient
    skyColor -= 0.5 * vec3(0.90, 0.75, 0.90) * rayDir.y;

    skyColor = skyColor * (1.0 - cloud.a) + cloud.rgb;
    outFragColor = vec4(skyColor, 1.f);
}

vec4 rayMarchVelocity(vec3 rayOrigin, vec3 rayDir) {
    vec4 final_color = vec4(0.0);

    float tNear, tFar;
    if (!intersectBox(rayOrigin, rayDir, tNear, tFar)) {
        return vec4(0.0);
    }

    float t = max(0.0, tNear);
    vec3 pos = rayOrigin + rayDir * t;

    for (int i = 0; i < MAX_STEPS && t < tFar; i++) {
        // Sample the full vec3 velocity vector
        vec3 velocity = texture(readOnlyTexs[ubo.texId], vec3(pos.x, 1.0 - pos.y, pos.z)).rgb;

        // Calculate magnitude to use as density/opacity
        // 3x is to see the texture field a bit clearer.
        float magnitude = 3.f * length(velocity);

        if (magnitude > 0.001) {
            // 1. Calculate opacity based on velocity magnitude
            float srcA = clamp(magnitude * INTENSITY_SCALE * ubo.rayStepSize, 0.0, 1.0);

            // 2. Map velocity direction to color
            // Option A: Use the absolute direction for RGB (X=R, Y=G, Z=B)
            // We use abs() or (v*0.5+0.5) to ensure values are in [0,1] range
            vec3 srcRGB = (normalize(velocity) + vec3(1)) * 0.5f;

            // Standard front-to-back alpha blending
            float weight = 1.0 - final_color.a;
            final_color.rgb += weight * srcRGB * srcA;
            final_color.a += weight * srcA;

            if (final_color.a >= 0.95) break;
        }

        t += ubo.rayStepSize;
        pos = rayOrigin + rayDir * t;
    }

    return final_color;
}

vec4 rayMarchScalar(vec3 rayOrigin, vec3 rayDir) {
    vec4 final_color = vec4(0.0);

    // Calculate proper entry and exit points
    float tNear, tFar;
    if (!intersectBox(rayOrigin, rayDir, tNear, tFar)) {
        // Ray completely misses the volume
        return vec4(0.0);
    }

    // Start at entry point (or origin if already inside)
    float t = max(0.0, tNear);
    vec3 pos = rayOrigin + rayDir * t;

    // March from entry to exit
    for (int i = 0; i < MAX_STEPS && t < tFar; i++) {
        // Sample temperature from your 3D texture (Binding 3, Index 5)
        float scalar = texture(readOnlyTexs[ubo.texId], vec3(pos.x, 1.0 - pos.y, pos.z)).r;

        if (scalar > 0.001) {
            float srcA = clamp(scalar * INTENSITY_SCALE * ubo.rayStepSize, 0.0, 1.0);

            vec3 srcRGB;
            if (ubo.texId == 5) {
                // Temperature
                // Map the normalized scalar to a thermal color palette
                srcRGB = INTENSITY_SCALE * getBlackbodyColor(clamp(scalar, 0.0, 1.0));
            } else if (ubo.texId == 1) {
                // Pressure
                float pScalar = abs(scalar);
                srcA = clamp(pScalar * INTENSITY_SCALE * ubo.rayStepSize, 0.0, 1.0);
                srcRGB = PRESSURE_COLOR;
            } else {
                // Smoke density
                srcRGB = mix(vec3(1.0, 1.0, 1.0), vec3(0.0, 0.0, 0.0), scalar);
            }

            float weight = 1.0 - final_color.a;
            final_color.rgb += weight * srcRGB * srcA;
            final_color.a += weight * srcA;

            if (final_color.a >= 0.95) break;
        }

        t += ubo.rayStepSize;
        pos = rayOrigin + rayDir * t;

    }

    return final_color;
}

vec3 getBlackbodyColor(float t) {
    // Standard "Ironbow" style thermal gradient
    // t should be normalized 0.0 to 1.0
    vec3 color;
    color.r = clamp(smoothstep(0.0, 0.3, t), 0.0, 1.0);
    color.g = clamp(smoothstep(0.3, 0.7, t), 0.0, 1.0);
    color.b = clamp(smoothstep(0.7, 1.0, t), 0.0, 1.0) + (smoothstep(0.0, 0.2, t) - smoothstep(0.2, 0.5, t));

    // Manual mapping for a standard thermal look:
    // Dark Blue (0) -> Purple -> Red -> Orange -> Yellow -> White (1)
    vec3 c1 = vec3(0.0, 0.0, 0.05);// Black/Blue
    vec3 c2 = vec3(0.5, 0.0, 0.5);// Purple
    vec3 c3 = vec3(1.0, 0.1, 0.0);// Red/Orange
    vec3 c4 = vec3(1.0, 0.9, 0.0);// Yellow
    vec3 c5 = vec3(1.0, 1.0, 1.0);// White

    if (t < 0.25) return mix(c1, c2, t * 4.0);
    if (t < 0.5)  return mix(c2, c3, (t - 0.25) * 4.0);
    if (t < 0.75) return mix(c3, c4, (t - 0.5) * 4.0);
    return mix(c4, c5, (t - 0.75) * 4.0);
}

bool intersectBox(vec3 rayOrigin, vec3 rayDir, out float tNear, out float tFar) {
    // Intersect ray with [0,1]³ volume
    vec3 invDir = 1.0 / rayDir;
    vec3 tMin = (vec3(0.0) - rayOrigin) * invDir;
    vec3 tMax = (vec3(1.0) - rayOrigin) * invDir;

    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);

    tNear = max(max(t1.x, t1.y), t1.z);
    tFar = min(min(t2.x, t2.y), t2.z);

    return tFar >= tNear && tFar >= 0.0;
}

float sdSphere(vec3 pos, vec3 center, float radius) {
    return length(pos - center) - radius;
}

float scene(vec3 pos) {
    float distance = sdSphere(pos, vec3(0.5, 0.5, 0.5), CLOUD_SIZE);

    float f = fbm(pos);

    // Negate to make interior positive (density)
    return -distance + f;
}

vec4 rayMarchSDF(vec3 rayOrigin, vec3 rayDir) {
    float tNear, tFar;
    if (!intersectBox(rayOrigin, rayDir, tNear, tFar)) {
        return vec4(0.0);// Ray misses volume
    }

    float t = max(0.0, tNear);
    vec3 pos = rayOrigin + rayDir * t;
    vec4 res = vec4(0.0);

    for (int i = 0; i < MAX_STEPS; i++) {
        float density = scene(pos);

        // We only draw the density if it's greater than 0
        if (density > 0.0) {
            vec4 color = vec4(mix(vec3(1.0, 1.0, 1.0), vec3(0.0, 0.0, 0.0), density), density);
            color.rgb *= color.a;
            res += color*(1.0-res.a);
        }

        t += ubo.rayStepSize;
        pos = rayOrigin + t * rayDir;
    }

    return res;
}

// Fractal Brownian Motion
float fbm(vec3 pos) {
    vec3 q = pos +  ubo.time * 0.5 * vec3(1.0, -0.2, -1.0);
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < 5; i++) {
        float noiseRes = snoise(q);
        value += amplitude * noiseRes;
        frequency *= 2.0;
        q *= 2;
        amplitude *= 0.5;
    }
    return float(value);
}

///----
/// Simplex 3D Noise
/// by Ian McEwan, Ashima Arts
vec4 permute(vec4 x) {
    return mod(((x * 34.0) + 1.0) * x, 289.0);
}

vec4 taylorInvSqrt(vec4 r) {
    return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 pos) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i = floor(pos + dot(pos, C.yyy));
    vec3 x0 = pos - i + dot(i, C.xxx);

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    //  x0 = x0 - 0. + 0.0 * C
    vec3 x1 = x0 - i1 + 1.0 * C.xxx;
    vec3 x2 = x0 - i2 + 2.0 * C.xxx;
    vec3 x3 = x0 - 1. + 3.0 * C.xxx;

    // Permutations
    i = mod(i, 289.0);
    vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y +
    vec4(0.0, i1.y, i2.y, 1.0)) +
    i.x + vec4(0.0, i1.x, i2.x, 1.0));

    // Gradients
    // ( N*N points uniformly over a square, mapped onto an octahedron.)
    float n_ = 1.0 / 7.0;// N=7
    vec3 ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);//  mod(p,N*N)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);// mod(j,N)

    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);

    // Normalise gradients
    vec4 norm =
    taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m =
    max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    return 42.0 *
    dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}
