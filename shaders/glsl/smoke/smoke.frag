#version 450

layout (location = 0) in vec2 outUV;

layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBOView
{
	mat4 cameraView;
    vec3 cameraPos;
    vec2 screenRes;
    float time;
} ubo;


const float MARCH_SIZE = 0.05;
const float MAX_STEPS = 200;
const vec3 SKY_COLOR = vec3(0.7, 0.7, 0.90);
const vec3 SUNLIGHT_DIRECTION = vec3(1.0, 0.0, 0.0);
const vec3 SUN_COLOR = vec3(1., .6, .0);
const float EPSILON = 0.0001;

vec4 permute(vec4 x);
vec4 taylorInvSqrt(vec4 r);
float snoise(vec3 pos);
float fbm(vec3 pos);
float sdCube(vec3 p, float cube_dim);
float sdSphere(vec3 pos, float radius);
vec4 rayMarch(vec3 rayOrigin, vec3 rayDirection);

void main() {
    // gl_FragCoord is in window coordinates where (0,0) is bottom-left
    // xy is in NDC [-1, +1].
    vec2 xy = 2*(gl_FragCoord.xy / ubo.screenRes.xy - vec2(0.5));
    // Aspect ratio correction for non-square screens
	xy.x *= ubo.screenRes.x / ubo.screenRes.y;
 
    // Extrapolate a 3D vector from the camera to the screen.
    // The camera is behind screen at (0, 0, -z) then (0, 0, 1) would be
    // center of the screen (near plane of frustum)
	vec3 dir = normalize(vec3(xy.x, xy.y, 1.0));
	dir = mat3(ubo.cameraView) * dir;

    // Keeps camera focal point at the center of screen
    vec3 newCameraPos = mat3(ubo.cameraView) * ubo.cameraPos;

    vec3 color = vec3(0.0);

	// Sun and Sky
	float sun = clamp(dot(SUNLIGHT_DIRECTION, dir), 0.0, 1.0 );
	// Base sky color
	color = SKY_COLOR;
	// Add sun color to sky
	color += 0.5 * SUN_COLOR * pow(sun, 15.0);

    // Cloud
    vec4 res = rayMarch(newCameraPos, dir);
    color = color * (1.0 - res.a) + res.rgb;

	outFragColor = vec4(color, 1.0);
}

// returns positive if point is outside sphere, negative inside
float sdSphere(vec3 p, float radius) {
  return length(p) - radius;
}

float sdCube(vec3 pos, float cube_dim) {
    vec3 q = abs(pos) - vec3(cube_dim / 2.f);
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Fractal Brownian Motion
float fbm(vec3 pos) {
  vec3 q = pos + ubo.time * 0.5 * vec3(1.0, -0.2, -1.0);

  float result = 0.0;
  float scale = 0.5;
  float factor = 2.02;

  for (int i = 0; i < 6; i++) {
      result += scale * snoise(q);
      q *= factor;
      factor += 0.21;
      scale *= 0.5;
  }

  return result;
}

float scene(vec3 pos) {
  float distance = sdSphere(pos, 1.0);

  float f = fbm(pos);

  return -distance + f;
}

vec4 rayMarch(vec3 rayOrigin, vec3 rayDirection) {
  float depth = 0.0;
  vec3 pos = rayOrigin + depth * rayDirection;

  vec4 res = vec4(0.0);

  for (int i = 0; i < MAX_STEPS; i++) {
    // +0.3 adds more base density to cloud
    float density = scene(pos) + 0.3;

    // We only draw the density if it's greater than 0
    if (density > 0.0) {
        // high to low density == low occlusion == more diffusion
        // low to high density == high occlusion == less diffusion
        float diffuse = clamp((scene(pos) - scene(pos + 0.3 * SUNLIGHT_DIRECTION)) / 0.3, 0.0, 1.0 );
        // interpolate sky color and sun color
        vec3 lerp = SKY_COLOR * 1.1 + 0.6 * SUN_COLOR * diffuse;
        // Smoke color (white to black) interpolated by density
        vec4 color = vec4(mix(vec3(1.0, 1.0, 1.0), vec3(0.0, 0.0, 0.0), density), density);
        color.rgb *= lerp;
        color.rgb *= color.a;
        res += color * (1.0 - res.a);
    }

    depth += MARCH_SIZE;
    pos = rayOrigin + depth * rayDirection;
  }

  return res;
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


