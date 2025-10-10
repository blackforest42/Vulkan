#version 450

layout (binding = 0) uniform UBO 
{
    // Tonemapping
	int tonemapEnabled;
    float exposure;

	// Bloom
	float bloomStrength;
} ubo;

layout (binding = 1) uniform sampler2D srcTexture;
layout (binding = 2) uniform sampler2D bloomTexture;

// in 
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

void main() {
	//outFragColor = vec4(0, 0, 1.f, 1.0);
	//return;

	vec3 srcColor = texture(srcTexture, inUV).rgb;
	vec3 bloomColor = texture(bloomTexture, inUV).rgb;

	outFragColor.rgb = bloomColor;
	return;

	// linear interpolation
	vec3 result = mix(srcColor, bloomColor, ubo.bloomStrength);

	if (ubo.tonemapEnabled == 1) {
		// Tonemapping
		vec3 fragColor = vec3(1.0) - exp(-result * ubo.exposure);
		// No need to gamma correct. sRGB swapchain auto gamma corrects
		//fragColor = pow(fragColor, vec3(1.0 / ubo.gamma));
		outFragColor.rgb = fragColor;
	} else {
		outFragColor.rgb = result;
	}

}