#version 450

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBO
{
    mat4 cameraView;
    vec3 cameraPos;
    vec2 resolution;

    float time;

    // Tonemapping
    float exposure;
    float gamma;

    int showBlackhole;
} ubo;

// Texture maps
layout (binding = 1) uniform samplerCube galaxyCubemap;
layout (binding = 2) uniform sampler2D colorMap;

const float PI = 3.14159265359;
const float EPSILON = 0.0001;
const float INFINITY = 1000000.0;
const int SAMPLES_PER_PIXEL = 300;

const bool frontView = false;
const bool topView = false;
const float cameraRoll = 0.0;

const float gravatationalLensing = 1.0;
const float fovScale = 1.0;

const float AccDiskEnabled = 1.0;
const float AccDiskParticle = 1.0;
const float AccDiskHeight = 0.55;
const float AccDiskLit = 0.25;
const float AccDiskDensityV = 2.0;
const float AccDiskDensityH = 4.0;
const float AccDiskNoiseScale = .8;
const float AccDiskNoiseLOD = 5.0;
const float AccDiskSpeed = 0.5;

// For debugging
struct Ring {
  vec3 center;
  vec3 normal;
  float innerRadius;
  float outerRadius;
  float rotateSpeed;
};

#define IN_RANGE(x, a, b) (((x) > (a)) && ((x) < (b)))

vec4 permute(vec4 x);
vec4 taylorInvSqrt(vec4 r);
float snoise(vec3 v);
float ringDistance(vec3 rayOrigin, vec3 rayDir, Ring ring);
vec3 accel(float h2, vec3 pos);
vec4 quadFromAxisAngle(vec3 axis, float angle);
vec4 quadConj(vec4 q);
vec4 quat_mult(vec4 q1, vec4 q2);
vec3 rotateVector(vec3 position, vec3 axis, float angle);
vec3 toSpherical(vec3 p);
void ringColor(vec3 rayOrigin,
               vec3 rayDir,
               Ring ring,
               inout float minDistance,
               inout vec3 color);
float sqrLength(vec3 a);
void accDiskColor(vec3 pos, inout vec3 color, inout float alpha);
vec3 traceColor(vec3 pos, vec3 dir);

void main() {
	// for testing purposes
	// outFragColor = texture(colorMap, inUV);
	// return;

    vec2 uv = gl_FragCoord.xy / ubo.resolution.xy - vec2(0.5);
	uv.x *= ubo.resolution.x / ubo.resolution.y;
 
	vec3 dir = normalize(vec3(-uv.x, -uv.y,  1.0));
	dir = mat3(ubo.cameraView) * dir;
    vec3 newCameraPos = mat3(ubo.cameraView) * ubo.cameraPos;
 
    vec3 hdrColor;
    if (ubo.showBlackhole == 1) {
        hdrColor = traceColor(newCameraPos, dir).rgb;
    } else {
        hdrColor = texture(galaxyCubemap, dir).rgb;
    }
    // Tonemapping
    vec3 mapped = vec3(1.0) - exp(-hdrColor * ubo.exposure);
	// Gamma correction
	outFragColor.rgb = pow(mapped, vec3(1.0 / ubo.gamma));
}


///////////////////////////////////////////////////////////////////////

///----
/// Simplex 3D Noise
/// by Ian McEwan, Ashima Arts
vec4 permute(vec4 x) {
  return mod(((x * 34.0) + 1.0) * x, 289.0);
}

vec4 taylorInvSqrt(vec4 r) {
  return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  // First corner
  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);

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

float ringDistance(vec3 rayOrigin, vec3 rayDir, Ring ring) {
  float denominator = dot(rayDir, ring.normal);
  float constant = -dot(ring.center, ring.normal);
  if (abs(denominator) < EPSILON) {
    return -1.0;
  } else {
    float t = -(dot(rayOrigin, ring.normal) + constant) / denominator;
    if (t < 0.0) {
      return -1.0;
    }

    vec3 intersection = rayOrigin + t * rayDir;

    // Compute distance to ring center
    float d = length(intersection - ring.center);
    if (d >= ring.innerRadius && d <= ring.outerRadius) {
      return t;
    }
    return -1.0;
  }
}

vec3 accel(float h2, vec3 pos) {
  float r2 = dot(pos, pos);
  float r5 = pow(r2, 2.5);
  vec3 acc = -1.5 * h2 * pos / r5 * 1.0;
  return acc;
}

vec4 quadFromAxisAngle(vec3 axis, float angle) {
  vec4 qr;
  float half_angle = (angle * 0.5) * 3.14159 / 180.0;
  qr.x = axis.x * sin(half_angle);
  qr.y = axis.y * sin(half_angle);
  qr.z = axis.z * sin(half_angle);
  qr.w = cos(half_angle);
  return qr;
}

vec4 quadConj(vec4 q) {
  return vec4(-q.x, -q.y, -q.z, q.w);
}

vec4 quat_mult(vec4 q1, vec4 q2) {
  vec4 qr;
  qr.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
  qr.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
  qr.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
  qr.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
  return qr;
}

vec3 rotateVector(vec3 position, vec3 axis, float angle) {
  vec4 qr = quadFromAxisAngle(axis, angle);
  vec4 qr_conj = quadConj(qr);
  vec4 q_pos = vec4(position.x, position.y, position.z, 0);

  vec4 q_tmp = quat_mult(qr, q_pos);
  qr = quat_mult(q_tmp, qr_conj);

  return vec3(qr.x, qr.y, qr.z);
}

#define IN_RANGE(x, a, b) (((x) > (a)) && ((x) < (b)))


// Convert from Cartesian to spherical coord (rho, phi, theta)
// https://en.wikipedia.org/wiki/Spherical_coordinate_system
vec3 toSpherical(vec3 p) {
  float rho = sqrt((p.x * p.x) + (p.y * p.y) + (p.z * p.z));
  float theta = atan(p.z, p.x);
  float phi = asin(p.y / rho);
  return vec3(rho, theta, phi);
}

void ringColor(vec3 rayOrigin,
               vec3 rayDir,
               Ring ring,
               inout float minDistance,
               inout vec3 color) {
  float distance = ringDistance(rayOrigin, normalize(rayDir), ring);
  if (distance >= EPSILON && distance < minDistance &&
      distance <= length(rayDir) + EPSILON) {
    minDistance = distance;

    vec3 intersection = rayOrigin + normalize(rayDir) * minDistance;
    vec3 ringColor;

    {
      float dist = length(intersection);

      float v = clamp(
          (dist - ring.innerRadius) / (ring.outerRadius - ring.innerRadius),
          0.0, 1.0);

      vec3 base = cross(ring.normal, vec3(0.0, 0.0, 1.0));
      float angle = acos(dot(normalize(base), normalize(intersection)));
      if (dot(cross(base, intersection), ring.normal) < 0.0)
        angle = -angle;

      float u = 0.5 - 0.5 * angle / PI;
      // HACK
      u += ubo.time * ring.rotateSpeed;

      vec3 color = vec3(0.0, 0.5, 0.0);
      // HACK
      float alpha = 0.5;
      ringColor = vec3(color);
    }

    color += ringColor;
  }
}

float sqrLength(vec3 a) {
  return dot(a, a);
}

void accDiskColor(vec3 pos, inout vec3 color, inout float alpha) {
  float innerRadius = 2.6;
  float outerRadius = 12.0;

  // Density linearly decreases as the distance to the blackhole center
  // increases.
  float density = max(
      0.0, 1.0 - length(pos.xyz / vec3(outerRadius, AccDiskHeight, outerRadius)));
  if (density < 0.001) {
    return;
  }

  density *= pow(1.0 - abs(pos.y) / AccDiskHeight, AccDiskDensityV);

  // Set particale density to 0 when radius is below the inner most stable
  // circular orbit.
  density *= smoothstep(innerRadius, innerRadius * 1.1, length(pos));

  // Avoid the shader computation when density is very small.
  if (density < 0.001) {
    return;
  }

  vec3 sphericalCoord = toSpherical(pos);

  // Scale the rho and phi so that the particales appear to be at the correct
  // scale visually.
  sphericalCoord.y *= 2.0;
  sphericalCoord.z *= 4.0;

  density *= 1.0 / pow(sphericalCoord.x, AccDiskDensityH);
  density *= 16000.0;

  if (AccDiskParticle < 0.5) {
    color += vec3(0.0, 1.0, 0.0) * density * 0.02;
    return;
  }

  float noise = 1.0;
  for (int i = 0; i < int(AccDiskNoiseLOD); i++) {
    noise *= 0.5 * snoise(sphericalCoord * pow(i, 2) * AccDiskNoiseScale) + 0.5;
    if (i % 2 == 0) {
      sphericalCoord.y += ubo.time * AccDiskSpeed;
    } else {
      sphericalCoord.y -= ubo.time * AccDiskSpeed;
    }
  }

  vec3 dustColor =
      texture(colorMap, vec2(sphericalCoord.x / outerRadius, 0.5)).rgb;

  color += density * AccDiskLit * dustColor * alpha * abs(noise);
}

vec3 traceColor(vec3 pos, vec3 dir) {
  vec3 color = vec3(0.0);
  float alpha = 1.0;

  float STEP_SIZE = 0.1;
  dir *= STEP_SIZE;

  // Initial values
  vec3 h = cross(pos, dir);
  float h2 = dot(h, h);

  for (int i = 0; i < SAMPLES_PER_PIXEL; i++) {
      // If gravatational lensing is applied
      if (gravatationalLensing > 0.5) {
        vec3 acc = accel(h2, pos);
        dir += acc;
      }

      // Reach event horizon
      if (dot(pos, pos) < 1.0) {
        return color;
      }

      float minDistance = INFINITY;

      // For debugging
      if (false) {
        Ring ring;
        ring.center = vec3(0.0, 0.05, 0.0);
        ring.normal = vec3(0.0, 1.0, 0.0);
        ring.innerRadius = 2.0;
        ring.outerRadius = 6.0;
        ring.rotateSpeed = 0.08;
        ringColor(pos, dir, ring, minDistance, color);
      } else {
        if (AccDiskEnabled > 0.5) {
          accDiskColor(pos, color, alpha);
        }
      }
    pos += dir;
  }

  // Sample skybox color
  dir = rotateVector(dir, vec3(0.0, 1.0, 0.0), ubo.time);
  color += texture(galaxyCubemap, dir).rgb * alpha;
  return color;
}

