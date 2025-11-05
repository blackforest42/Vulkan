#version 450

// in
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTranslation;

// out
layout(location = 0) out vec2 outUV;

layout (binding = 1) uniform sampler2D velocityMap;

mat2 rot(float angle) {
	float c = cos(angle);
	float s = sin(angle);
	return mat2(vec2(c, -s), vec2(s, c));
}


void main() {
    outUV = (inTranslation.xy + 1.0) * 0.5;
    vec2 velocity = texture(velocityMap, outUV).xy;
    float scale = 0.05 * length(velocity);
    float angle = atan(velocity.y, velocity.x);
    mat2 rotation = rot(-angle);
              
    gl_Position = vec4(rotation * (scale * inPosition) + inTranslation, 0, 1);
}