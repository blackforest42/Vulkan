#version 450

// out
layout (location = 0) out vec4 outFragColor;

// layout (binding = 1) uniform samplerCube samplerCubeMap;

const float PI = 3.14159265359;
const float EPSILON = 0.0001;
const float INFINITY = 1000000.0;

const bool frontView = false;
const bool topView = false;
const float cameraRoll = 0.0;

const float gravatationalLensing = 1.0;
const float renderBlackHole = 1.0;
const float mouseControl = 1.0;
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
	outFragColor = vec4(vec3(0.f, 0.f, 1.f), 1.f);
}
