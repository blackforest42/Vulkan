#version 450

#extension GL_EXT_debug_printf : enable


// ins
layout (location = 0) in vec2 uv;

// outs
layout (location = 0) out vec4 outFragColor;

layout (binding = 0) uniform UBO  {
    vec2 dstResolution;
} ubo;

layout (binding = 1) uniform sampler2D srcTexture;

void main() {
    //outFragColor = texture(srcTexture, uv);
    //debugPrintfEXT("Viewport resolution : %f, %f", ubo.dstResolution.x, ubo.dstResolution.y);
    vec2 inputTexelSize = 1.0 / ubo.dstResolution * 0.5;
    vec4 o = inputTexelSize.xyxy * vec4(-1.0, -1.0, 1.0, 1.0); // Offset
        outFragColor =
            0.25 * (texture(srcTexture, uv + o.xy) + texture(srcTexture, uv + o.zy) +
                texture(srcTexture, uv + o.xw) + texture(srcTexture, uv + o.zw));
}