#version 450

layout (binding = 0) uniform UBO 
{
    // Tonemapping
	int tonemapEnabled;
    float exposure;

	// Bloom
	float bloomStrength;

	// debug
	int debug;
} ubo;

layout (binding = 1) uniform sampler2D blackholeTex;
layout (binding = 2) uniform sampler2D upSampledTex;
layout (binding = 3) uniform sampler2D debugTex;

// in 
layout (location = 0) in vec2 inUV;

// out
layout (location = 0) out vec4 outFragColor;

void main() {
	//outFragColor = vec4(0, 0, 1.f, 1.0);
	//return;

	vec3 blackholeColor = texture(blackholeTex, inUV).rgb;
	vec3 upSampledColor = texture(upSampledTex, inUV).rgb;

	// Debug
	if (ubo.debug == 1) {
		outFragColor.rgb = texture(debugTex, inUV).rgb;
		return;
	}

	// linear interpolation
	// vec3 result = mix(blackholeColor, upSampledColor, ubo.bloomStrength);
	// Orignal mixing procedure
	vec3 result = blackholeColor + upSampledColor * ubo.bloomStrength;

	if (ubo.tonemapEnabled == 1) {
		// Tonemapping
		vec3 fragColor = vec3(1.0) - exp(-result * ubo.exposure);
		// No need to gamma correct. sRGB swapchain auto gamma corrects
		fragColor = pow(fragColor, vec3(1.0 / 2.2f));
		outFragColor.rgb = fragColor;
	} else {
		outFragColor.rgb = result;
	}

}