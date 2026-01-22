#version 450

// in
layout(location = 0) in vec2 inUV;

// out
layout(location = 0) out vec4 outFragColor;

layout(binding = 0) uniform RayMarchUBO {
  mat4 model;
  mat4 invModel;
  mat4 cameraView;
  mat4 perspective;
  vec3 cameraPos;
  vec2 screenRes;
  float time;
  int toggleView;
}
ubo;
layout(binding = 1) uniform sampler3D volumeTexture;
layout(binding = 2) uniform sampler2D preMarchFrontTex;
layout(binding = 3) uniform sampler2D preMarchBackTex;

const float STEP_SIZE = 0.01;
const float MAX_STEPS = 200;
const float SMOKE_DENSITY = 1.0f;
const float CLOUD_SIZE = .5f;
const vec3 SMOKE_COLOR = vec3(1.0);
const float DENSITY_SCALE = 1.0f;

vec4 permute(vec4 x);
vec4 taylorInvSqrt(vec4 r);
float snoise(vec3 pos);
float fbm(vec3 pos);
vec4 rayMarch(vec3 rayOrigin, vec3 rayDirection);
vec4 rayMarchNoise(vec3 rayOrigin, vec3 rayDir);
vec4 rayMarchSDF(vec3 rayOrigin, vec3 rayDir);
bool intersectBox(vec3 rayOrigin, vec3 rayDir, out float tNear, out float tFar);

void main() {
    vec4 entry = texture(preMarchBackTex, inUV);
    // Check if ray is occluded (see PS_RAYDATA_FRONT)
    if (entry.w == 0.0f) {
        discard;
    }
    outFragColor = vec4(1.f);

/*
	// Ray direction from camera to this point on the cube
	vec3 worldPos = inPos;
	vec3 rayDir = worldPos - ubo.cameraPos;
	// Transform ray to texture space [0, 1]
	vec3 rayDirTexSpace = normalize(mat3(ubo.invModel) * rayDir);
	// Flip the texture since it is upside down
	vec3 uvw = inUVW;
	uvw.y = 1 - inUVW.y;

	vec4 color;
	// March rays starting from front face of cube
	if (ubo.toggleView == 0) {
	    color = rayMarch(uvw, rayDirTexSpace);
	} else if (ubo.toggleView == 1) {
	    // color = rayMarchNoise(uvw, rayDirTexSpace);
	    color = rayMarchSDF(uvw, rayDirTexSpace);
	}
    else if (ubo.toggleView == 2) {
		outFragColor = vec4(rayDirTexSpace * 0.5 + 0.5, 1.0);
		return;
    }
	outFragColor = vec4(color);
*/
}

vec4 rayMarch(vec3 rayOrigin, vec3 rayDir) {
    vec4 final_color = vec4(0.0);
    
    // Calculate proper entry and exit points
    float tNear, tFar;
    if (!intersectBox(rayOrigin, rayDir, tNear, tFar)) {
        return vec4(0.0);  // Ray completely misses the volume
    }
    
    // Start at entry point (or origin if already inside)
    float t = max(0.0, tNear);
    vec3 pos = rayOrigin + rayDir * t;
    
    // March from entry to exit
    for (int i = 0; i < MAX_STEPS && t < tFar; i++) {
        float density = texture(volumeTexture, pos).r;
        
        if (density > 0.001) {  // Small threshold to skip empty space
            float srcA = clamp(density * DENSITY_SCALE * STEP_SIZE, 0.0, 1.0);
            vec3 srcRGB = SMOKE_COLOR * srcA;
            
            float weight = 1.0 - final_color.a;
            final_color.rgb += weight * srcRGB;
            final_color.a += weight * srcA;
            
            if (final_color.a >= 0.95) break;
        }
        
        // Step along ray parametrically
        t += STEP_SIZE;
        pos = rayOrigin + rayDir * t;
    }
    
    return final_color;
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
        return vec4(0.0);  // Ray misses volume
    }
    
    float t = max(0.0, tNear);
    vec3 pos = rayOrigin + rayDir * t;
  vec4 res = vec4(0.0);

  for (int i = 0; i < MAX_STEPS; i++) {
    float density = scene(pos);

    // We only draw the density if it's greater than 0
    if (density > 0.0) {
      vec4 color = vec4(mix(vec3(1.0,1.0,1.0), vec3(0.0, 0.0, 0.0), density), density );
      color.rgb *= color.a;
      res += color*(1.0-res.a);
    }

    t += STEP_SIZE;
    pos = rayOrigin + t * rayDir;
  }

  return res;
}

vec4 rayMarchNoise(vec3 rayOrigin, vec3 rayDir) {
	float t = 0.00;
	vec3 pos = rayOrigin + rayDir * t;
	float density = 0.f;
	float totalDensity = 0.0;
	// Animated position for smoke movement
	vec3 windOffset = vec3(ubo.time * 0.1, ubo.time * 0.15, ubo.time * 0.08);

	for (int i = 0; i < MAX_STEPS; i++) {
		// Sample 3D noise texture with animation
		vec3 samplePos = pos * 1.5 + windOffset;
		float noise = fbm(samplePos);

		// Create smoke shape (spherical falloff)
		float dist = length(pos - vec3(0.5));
		float falloff = smoothstep(1.5 * CLOUD_SIZE, 0.5 * CLOUD_SIZE, dist);

		// Combine noise with falloff
		float smokeDensity = max(0.0, noise * 0.5 + 0.5) * falloff * SMOKE_DENSITY;

		// Accumulate density
		totalDensity += smokeDensity * STEP_SIZE;

		// Add lighting based on position
		vec3 lightDir = normalize(vec3(1.0, 1.0, -0.5));
		float lighting = max(0.3, dot(normalize(pos), lightDir) * 0.5 + 0.5);

		t += STEP_SIZE;
		pos = rayOrigin + rayDir * t;
	}

	float alpha = clamp(totalDensity, 0.0, 1.0);
	return vec4(SMOKE_COLOR, alpha);
}

// Fractal Brownian Motion
float fbm(vec3 pos) {
  vec3 q = pos + ubo.time * 0.5 * vec3(1.0, -0.2, -1.0);
  float value = 0.0;
  float amplitude = 0.5;
  float frequency = 1.0;

  for (int i = 0; i < 5; i++) {
    value += amplitude * snoise(q * frequency);
    frequency *= 2.0;
    amplitude *= 0.5;
  }
  return value;
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
  float n_ = 1.0 / 7.0;  // N=7
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,N*N)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);  // mod(j,N)

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
