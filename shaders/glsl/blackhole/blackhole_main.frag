#version 450

// in
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBO
{
    mat4 cameraView;
    vec2 resolution;
    float time;
    bool mouseControl;
} ubo;

// Texture maps
layout (binding = 1) uniform samplerCube galaxyCubemap;
layout (binding = 2) uniform sampler2D colorMap;

const float PI = 3.14159265359;
const float EPSILON = 0.0001;
const float INFINITY = 1000000.0;

const bool frontView = false;
const bool topView = false;
const float cameraRoll = 0.0;

const float gravatationalLensing = 1.0;
const float renderBlackHole = 1.0;
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

struct Ring {
  vec3 center;
  vec3 normal;
  float innerRadius;
  float outerRadius;
  float rotateSpeed;
};

#define IN_RANGE(x, a, b) (((x) > (a)) && ((x) < (b)))

void main() {
	// for testing purposes
	// outFragColor = texture(colorMap, inUV);
	// return;

    vec2 uv = gl_FragCoord.xy / ubo.resolution.xy - vec2(0.5);
	uv.x *= ubo.resolution.x / ubo.resolution.y;
 
	vec3 dir = normalize(vec3(-uv.x, uv.y,  1.0));
	dir = mat3(ubo.cameraView) * dir;

	outFragColor = texture(galaxyCubemap, vec3(dir));
}

