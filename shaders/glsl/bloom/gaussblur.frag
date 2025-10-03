#version 450

layout (binding = 0) uniform UBO 
{
	float blurScale;
	float blurStrength;
    float exposure;
    float gamma;
} ubo;

layout (binding = 1) uniform sampler2D samplerColor;

layout (constant_id = 0) const int blurdirection = 0;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	float weight[5];
	weight[0] = 0.227027;
	weight[1] = 0.1945946;
	weight[2] = 0.1216216;
	weight[3] = 0.054054;
	weight[4] = 0.016216;

	// gets size of single texel
	vec2 tex_offset = 1.0 / textureSize(samplerColor, 0) * ubo.blurScale;
	// current fragment's contribution
	vec3 result = texture(samplerColor, inUV).rgb * weight[0];

	for(int i = 1; i < 5; ++i)
	{
		if (blurdirection == 1)
		{
			// Horizontal
			result += texture(samplerColor, inUV + vec2(tex_offset.x * i, 0.0)).rgb * weight[i] * ubo.blurStrength;
			result += texture(samplerColor, inUV - vec2(tex_offset.x * i, 0.0)).rgb * weight[i] * ubo.blurStrength;
		}
		else
		{
			// Vertical
			result += texture(samplerColor, inUV + vec2(0.0, tex_offset.y * i)).rgb * weight[i] * ubo.blurStrength;
			result += texture(samplerColor, inUV - vec2(0.0, tex_offset.y * i)).rgb * weight[i] * ubo.blurStrength;
		}
	}

	// Tonemapping
	vec3 fragColor = vec3(1.0) - exp(-result * ubo.exposure);
	// gamma correct
	fragColor = pow(fragColor, vec3(1.0 / ubo.gamma));

	outFragColor = vec4(fragColor, 1.0);
}