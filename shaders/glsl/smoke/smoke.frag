#version 450

layout (location = 0) in vec2 outUV;

layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBOView
{
	mat4 cameraView;
    vec3 cameraPos;
    vec2 screenRes;
} ubo;


const float MARCH_SIZE = 0.08;
const float MAX_STEPS = 1000;

float sdSphere(vec3 pos, float radius);
vec4 rayMarch(vec3 rayOrigin, vec3 rayDirection);

void main() 
{
    // gl_FragCoord is in window coordinates where (0,0) is bottom-left
    // xy is in NDC [-1, +1].
    vec2 xy = 2*(gl_FragCoord.xy / ubo.screenRes.xy - vec2(0.5));
    // Aspect ratio correction for non-square screens
	xy.x *= ubo.screenRes.x / ubo.screenRes.y;
 
    // Extrapolate a 3D vector from the camera to the screen.
    // The camera is behind screen at (0, 0, -z) then (0, 0, 1) would be
    // center of the screen (near plane of frustum)
	vec3 dir = normalize(vec3(xy.x, -xy.y,  1.0));
	dir = mat3(ubo.cameraView) * dir;

    // Keeps camera focal point at the center of screen
    vec3 newCameraPos = mat3(ubo.cameraView) * ubo.cameraPos;

    vec4 res = rayMarch(newCameraPos, dir);

	outFragColor = vec4(res.rgb, 1.0);
}

float sdSphere(vec3 pos, float radius) {
    return length(pos) - radius;
}

float scene(vec3 p) {
  float distance = sdSphere(p, 1.0);
  return -distance;
}

vec4 rayMarch(vec3 rayOrigin, vec3 rayDirection) {
  float depth = 0.0;
  vec3 p = rayOrigin + depth * rayDirection;

  vec4 res = vec4(0.0);

  for (int i = 0; i < MAX_STEPS; i++) {
    float density = scene(p);

    // We only draw the density if it's greater than 0
    if (density > 0.0) {
      vec4 color = vec4(mix(vec3(1.0,1.0,1.0), vec3(0.0, 0.0, 0.0), density), density );
      color.rgb *= color.a;
      res += color * (1.0 - res.a);
    }

    depth += MARCH_SIZE;
    p = rayOrigin + depth * rayDirection;
  }

  return res;
}
