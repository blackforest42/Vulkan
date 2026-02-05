#version 450

// in
layout (location = 0) in vec2 inUV[];

// out
layout (location = 0) out vec2 outUV;

layout (set = 0, binding = 0) uniform UBO
{
    mat4 perspective;
    mat4 view;
} ubo;

layout(quads, equal_spacing, cw) in;

void main()
{
    // Interpolate UV coordinates
    vec2 uv1 = mix(inUV[0], inUV[1], gl_TessCoord.x);
    vec2 uv2 = mix(inUV[3], inUV[2], gl_TessCoord.x);
    outUV = mix(uv1, uv2, gl_TessCoord.y);

    // Interpolate positions
    vec4 pos1 = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);
    vec4 pos2 = mix(gl_in[3].gl_Position, gl_in[2].gl_Position, gl_TessCoord.x);
    vec4 pos = mix(pos1, pos2, gl_TessCoord.y);

    gl_Position = ubo.perspective * ubo.view * pos;
}

/*
From Claude
// water.tese
#version 450

layout(quads, fractional_odd_spacing, ccw) in;

in vec2 tcTexCoord[];
out vec3 teWorldPos;
out vec2 teTexCoord;
out mat3 teTBN;  // Tangent-Bitangent-Normal matrix

uniform mat4 modelMatrix;
uniform mat4 viewProjMatrix;

// Wave parameters (same as compute shader)
layout(binding = 0) uniform WaveParams {
    vec4 frequency[4];
    // ... same structure
} waves;

vec3 gerstnerWave(vec2 pos, out vec3 tangent, out vec3 bitangent) {
    vec3 displacement = vec3(0.0);
    tangent = vec3(1.0, 0.0, 0.0);
    bitangent = vec3(0.0, 1.0, 0.0);

    // Calculate 4 primary geometric waves
    for(int i = 0; i < 4; i++) {
        float freq = waves.frequency[0][i];
        float amp = waves.amplitude[0][i];
        vec2 dir = waves.direction[i].xy;
        float phase = waves.phase[0][i];

        float wave = dot(dir, pos) * freq + phase;
        float sinW = sin(wave);
        float cosW = cos(wave);

        // Vertical displacement
        displacement.z += amp * sinW;

        // Horizontal displacement (chop)
        float chop = 2.5;  // Same as m_GeoState.m_Chop in original
        displacement.xy += dir * amp * cosW * chop;

        // Tangent space derivatives
        float wa = freq * amp;
        tangent.xy -= dir * dir.x * wa * sinW;
        tangent.z += dir.x * wa * cosW;

        bitangent.xy -= dir * dir.y * wa * sinW;
        bitangent.z += dir.y * wa * cosW;
    }

    return displacement;
}

void main() {
    // Bilinear interpolation of patch corners
    vec2 uv1 = mix(tcTexCoord[0], tcTexCoord[1], gl_TessCoord.x);
    vec2 uv2 = mix(tcTexCoord[3], tcTexCoord[2], gl_TessCoord.x);
    teTexCoord = mix(uv1, uv2, gl_TessCoord.y);

    // Base world position
    vec3 pos = vec3(teTexCoord * 100.0 - 50.0, 0.0);  // Scale to world space

    // Apply Gerstner waves
    vec3 tangent, bitangent;
    pos += gerstnerWave(pos.xy, tangent, bitangent);

    teWorldPos = (modelMatrix * vec4(pos, 1.0)).xyz;

    // Build TBN matrix for normal mapping
    vec3 normal = normalize(cross(tangent, bitangent));
    teTBN = mat3(normalize(tangent), normalize(bitangent), normal);

    gl_Position = viewProjMatrix * vec4(teWorldPos, 1.0);
}
*/
