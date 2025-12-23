#version 450

layout (location = 0) in vec3 inLookAt;

layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBOView
{
	mat4 projection;
	mat4 view;
    mat4 invModelView;
    vec3 cameraPos;
} uboView;


const float MARCH_SIZE = 0.08;
const float MAX_STEPS = 100;

float sdSphere(vec3 pos, float radius);
vec4 raymarch(vec3 rayOrigin, vec3 rayDirection);

void main() 
{
	vec3 lookAt = normalize(inLookAt);

    raymarch(uboView.cameraPos, lookAt);

	outFragColor = vec4(1.0, 0, 0, .3f);
}

float sdSphere(vec3 pos, float radius) {
    return length(pos) - radius;
}

float scene(vec3 p) {
  float distance = sdSphere(p, 1.0);
  return -distance;
}

vec4 raymarch(vec3 rayOrigin, vec3 rayDirection) {
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
