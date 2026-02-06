#version 450

// in
layout (location = 0) in vec2 inUV[];

// out
layout (vertices = 4) out;
layout (location = 0) out vec2 outUV[4];

layout(set = 0, binding = 0) uniform UBO
{
    mat4 perspective;
    mat4 view;
    vec3 camera_pos;
    vec2 screen_res;
    float pixelsPerEdge;
} ubo;

// Tessellation configuration
layout(set = 0, binding = 1) uniform TessellationConfig
{
    float minTessLevel;
    float maxTessLevel;
    float minDistance;
    float maxDistance;
    float frustumCullMargin;
} tessConfig;


const float EPSILON = 0.001;

float calculateTessLevel(vec3 worldPos);
float calculateEdgeTessLevel(vec3 worldPos0, vec3 worldPos1);
bool isInFrustum(vec4 clipSpacePos);
bool shouldCullPatch();

void main(void)
{
    // Only first invocation calculates tessellation levels
    if (gl_InvocationID == 0)
    {
        // --- Frustum Culling ---
        if (shouldCullPatch())
        {
            // Set tessellation levels to 0 to cull the patch
            gl_TessLevelOuter[0] = 0.0;
            gl_TessLevelOuter[1] = 0.0;
            gl_TessLevelOuter[2] = 0.0;
            gl_TessLevelOuter[3] = 0.0;
            gl_TessLevelInner[0] = 0.0;
            gl_TessLevelInner[1] = 0.0;
        }
        else
        {
            // --- Calculate Tessellation Levels ---

            // Get world space positions
            vec3 worldPos[4];
            worldPos[0] = gl_in[0].gl_Position.xyz;
            worldPos[1] = gl_in[1].gl_Position.xyz;
            worldPos[2] = gl_in[2].gl_Position.xyz;
            worldPos[3] = gl_in[3].gl_Position.xyz;

            // Calculate edge tessellation levels
            // Edge 0: vertex 0 -> vertex 2 (left edge in standard quad layout)
            // Edge 1: vertex 0 -> vertex 1 (bottom edge)
            // Edge 2: vertex 1 -> vertex 3 (right edge)
            // Edge 3: vertex 2 -> vertex 3 (top edge)

            float tessLevel0 = calculateEdgeTessLevel(worldPos[0], worldPos[2]);
            float tessLevel1 = calculateEdgeTessLevel(worldPos[0], worldPos[1]);
            float tessLevel2 = calculateEdgeTessLevel(worldPos[1], worldPos[3]);
            float tessLevel3 = calculateEdgeTessLevel(worldPos[2], worldPos[3]);

            // Set outer tessellation levels
            gl_TessLevelOuter[0] = tessLevel0;
            gl_TessLevelOuter[1] = tessLevel1;
            gl_TessLevelOuter[2] = tessLevel2;
            gl_TessLevelOuter[3] = tessLevel3;

            // Set inner tessellation levels
            // Use maximum of opposite edges to prevent cracks
            gl_TessLevelInner[0] = max(tessLevel1, tessLevel3);
            gl_TessLevelInner[1] = max(tessLevel0, tessLevel2);

            // Alternative: Use average for smoother transitions
            //            gl_TessLevelInner[0] = (tessLevel1 + tessLevel3) * 0.5;
            //            gl_TessLevelInner[1] = (tessLevel0 + tessLevel2) * 0.5;
        }
    }

    // Pass through data for all invocations
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    outUV[gl_InvocationID] = inUV[gl_InvocationID];
}

// Calculate distance-based tessellation level
float calculateTessLevel(vec3 worldPos)
{
    // Calculate distance from camera to vertex
    float distance = length(worldPos - ubo.camera_pos);

    // Normalize distance between min and max range
    float normalizedDistance = clamp(
    (distance - tessConfig.minDistance) /
    (tessConfig.maxDistance - tessConfig.minDistance),
    0.0, 1.0
    );

    // Apply smoothstep for smoother LOD transitions
    normalizedDistance = smoothstep(0.0, 1.0, normalizedDistance);

    // Interpolate between max and min tessellation levels
    return mix(tessConfig.maxTessLevel, tessConfig.minTessLevel, normalizedDistance);
}

// Calculate tessellation level for an edge based on both vertices
float calculateEdgeTessLevel(vec3 worldPos0, vec3 worldPos1)
{
    // Use the maximum tessellation level of the two vertices
    // This prevents cracks between patches
    float tess0 = calculateTessLevel(worldPos0);
    float tess1 = calculateTessLevel(worldPos1);

    // Take the maximum to ensure smooth transitions
    return max(tess0, tess1);
}

// Check if a point is inside the view frustum (approximate)
bool isInFrustum(vec4 clipSpacePos)
{
    // Add small margin to prevent popping at frustum edges
    float margin = tessConfig.frustumCullMargin;

    // Check all 6 frustum planes (simplified)
    return clipSpacePos.x >= -clipSpacePos.w - margin &&
    clipSpacePos.x <=  clipSpacePos.w + margin &&
    clipSpacePos.y >= -clipSpacePos.w - margin &&
    clipSpacePos.y <=  clipSpacePos.w + margin &&
    clipSpacePos.z >= -clipSpacePos.w - margin &&
    clipSpacePos.z <=  clipSpacePos.w + margin;
}

// Check if entire patch should be culled
bool shouldCullPatch()
{
    mat4 mvp = ubo.perspective * ubo.view;

    // Transform all 4 vertices to clip space
    vec4 clipPos[4];
    clipPos[0] = mvp * gl_in[0].gl_Position;
    clipPos[1] = mvp * gl_in[1].gl_Position;
    clipPos[2] = mvp * gl_in[2].gl_Position;
    clipPos[3] = mvp * gl_in[3].gl_Position;

    // If all vertices are outside any single frustum plane, cull the patch
    bool allOutside = true;

    // Check each frustum plane
    for (int i = 0; i < 4; i++)
    {
        if (isInFrustum(clipPos[i]))
        {
            allOutside = false;
            break;
        }
    }

    return allOutside;
}

