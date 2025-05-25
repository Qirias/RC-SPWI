#define METAL
#include <metal_stdlib>

using namespace metal;
using namespace raytracing;

#include "vertexData.hpp"
#include "shaderTypes.hpp"
#include "shaderCommon.hpp"
#include "common.hpp"

struct TriangleResources {
    struct TriangleData {
        float4 normals[3];
        float4 colors[3];
    };
    device TriangleData* triangles;
};

float4 sun(float3 rayDir, FrameData frameData) {
    float3 sunDirection = normalize(-frameData.sun_eye_direction.xyz);
    float3 sunColor = frameData.sun_color.rgb;
    
    float sunDot = dot(rayDir, sunDirection);
    const float sunSize = 0.97f;
    
    float sunDisk = (sunDot > sunSize) ? 1.0f : 0.0f;
    float intensity = frameData.sun_specular_intensity;
    
    return float4(sunColor * sunDisk * intensity, 1.0f);
}

float4 sky(float3 rayDir, FrameData frameData) {
    const float3 skyZenithColor = float3(0.0, 0.4, 0.8);
    const float3 skyHorizonColor = float3(0.3, 0.6, 0.8);
    
    float upDot = max(0.0f, rayDir.y);
    float3 skyGradient = mix(skyHorizonColor, skyZenithColor, pow(upDot, 0.5f));
    
    float skyIntensity = frameData.sun_specular_intensity * 1.0f;
    float skyMask = (rayDir.y > 0.0f) ? 1.0f : 0.0f;

    float3 finalSkyColor = skyGradient * skyIntensity;
    
    return float4(finalSkyColor * skyMask, 1.0f);
}

float4 skyAndSun(float3 rayDir, FrameData frameData, CascadeData cascadeData) {
    float4 result = float4(0.0f, 0.0f, 0.0f, 1.0f);
    
    if (cascadeData.enableSun) {
        result += sun(rayDir, frameData);
    }
    
    if (cascadeData.enableSky) {
        result += sky(rayDir, frameData);
    }
    
    return result;
}

float3 reconstructWorldPositionFromLinearDepth(float2 ndc, float linearDepth, FrameData frameData) {
    float4 clipPos = float4(ndc.x, ndc.y, -1.0, 1.0);
    float4 viewPos = frameData.projection_matrix_inverse * clipPos;
    viewPos /= viewPos.w;
    
    float scale = linearDepth / fabs(viewPos.z);
    float depthBias = 0.98; // depth bias to avoid acne

    float3 viewPosAtDepth = viewPos.xyz * (scale * depthBias);
    
    float4 worldPos = frameData.view_matrix_inverse * float4(viewPosAtDepth, 1.0);
    return worldPos.xyz;
}

// https://www.shadertoy.com/view/4XXSWS
// Project a point onto a line and return the parametric distance
float projectLinePerpendicular(float3 lineStart, float3 lineEnd, float3 point) {
    float3 line = lineEnd - lineStart;
    float lineLength2 = dot(line, line);
    
    return saturate(dot(point - lineStart, line) / lineLength2);
}

// Calculate 3D-aware bilinear ratios using iterative approach
float2 getBilinear3dRatioIter(float3 srcPoints[4], float3 dstPoint, float2 initRatio, int iterCount) {
    float2 ratio = initRatio;
    
    for (int i = 0; i < iterCount; i++) {
        // Interpolate along Y and find X ratio
        float3 mixedY1 = mix(srcPoints[0], srcPoints[2], ratio.y);
        float3 mixedY2 = mix(srcPoints[1], srcPoints[3], ratio.y);
        ratio.x = projectLinePerpendicular(mixedY1, mixedY2, dstPoint);
        
        // Interpolate along X and find Y ratio
        float3 mixedX1 = mix(srcPoints[0], srcPoints[1], ratio.x);
        float3 mixedX2 = mix(srcPoints[2], srcPoints[3], ratio.x);
        ratio.y = projectLinePerpendicular(mixedX1, mixedX2, dstPoint);
    }
    
    return ratio;
}

float4 mergeUpperCascade(texture2d<float, access::sample> upperRadianceTexture,
                         texture2d<float, access::sample> depthTexture,
                         float2 probeUV,
                         float3 rayDir,
                         float currentDepth,
                         float3 currentWorldPos,
                         CascadeData cascadeData,
                         FrameData frameData) {
    uint upperTexWidth = upperRadianceTexture.get_width();
    uint upperTexHeight = upperRadianceTexture.get_height();
    
    uint currentCascadeLevel = cascadeData.cascadeLevel;
    uint upperCascadeLevel = currentCascadeLevel + 1;
    
    // Calculate parameters for current cascade
    uint probeSpacing = cascadeData.probeSpacing;
    uint raysPerDim = (1 << (currentCascadeLevel + 2));
    
    // Calculate parameters for upper cascade
    uint upperTileSize = probeSpacing * (1 << upperCascadeLevel);
    uint upperProbeGridSizeX = (frameData.framebuffer_width + upperTileSize - 1) / upperTileSize;
    uint upperProbeGridSizeY = (frameData.framebuffer_height + upperTileSize - 1) / upperTileSize;
    
    float2 octDir = octEncode(rayDir);
    
    // Calculate ray indices and fractions for interpolation
    float2 dirCoord = octDir * float(raysPerDim);
    int2 dirBase = int2(floor(dirCoord));
    float2 dirFrac = fract(dirCoord);
    
    float4 dirBilinearWeights = float4((1.0f - dirFrac.x)   * (1.0f - dirFrac.y),
                                        dirFrac.x           * (1.0f - dirFrac.y),
                                        (1.0f - dirFrac.x)  * dirFrac.y,
                                        dirFrac.x           * dirFrac.y);
    
    // For 3D-aware interpolation, we need probe positions in upper cascade
    float2 upperProbeCoord = probeUV * float2(upperProbeGridSizeX, upperProbeGridSizeY) - 0.5f;
    int2 upperProbeBase = int2(floor(upperProbeCoord));
    float2 upperProbeFrac = fract(upperProbeCoord);
    int2 probeOffsets[4] = {int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1)};
    
    float4 upperProbeDepths;
    int2 probeCoords[4];
    float2 probeUVs[4];
    float3 probeWorldPos[4];
    
    for (int i = 0; i < 4; i++) {
        probeCoords[i] = clamp(upperProbeBase + probeOffsets[i], int2(0), int2(upperProbeGridSizeX - 1, upperProbeGridSizeY - 1));
                              
        probeUVs[i] = (float2(probeCoords[i]) + 0.5f) / float2(upperProbeGridSizeX, upperProbeGridSizeY);
        upperProbeDepths[i] = depthTexture.sample(depthSampler, probeUVs[i]).x;
        
        float2 probeNDC = probeUVs[i] * 2.0f - 1.0f;
        probeNDC.y = -probeNDC.y;
        probeWorldPos[i] = reconstructWorldPositionFromLinearDepth(probeNDC, upperProbeDepths[i], frameData);
    }
    
    // Calculate 3D-aware interpolation weights
    float2 ratio3d = getBilinear3dRatioIter(probeWorldPos, currentWorldPos, upperProbeFrac, 4);
    float4 bilinearWeights = float4((1.0f - ratio3d.x)  * (1.0f - ratio3d.y),
                                    ratio3d.x           * (1.0f - ratio3d.y),
                                    (1.0f - ratio3d.x)  * ratio3d.y,
                                    ratio3d.x           * ratio3d.y);
    
    int2 dirOffsets[4] = {int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1)};
    
    float4 accumulatedRadiance = float4(0.0f);
    
    for (int probeIdx = 0; probeIdx < 4; probeIdx++) {
        float probeWeight = bilinearWeights[probeIdx];
        if (probeWeight <= 0.0001f) continue;
        
        float4 probeRadiance = float4(0.0f);
        
        for (int dirIdx = 0; dirIdx < 4; dirIdx++) {
            int2 upperRayIndex = dirBase * 2 + dirOffsets[dirIdx];
            
            // For direction-first layout, the texture coordinates are:
            // (rayIndex * probeGridSize + probeIndex)
            // This puts the probe grid for each direction adjacent to each other
            float2 texelPos = float2(upperRayIndex) * float2(upperProbeGridSizeX, upperProbeGridSizeY) + float2(probeCoords[probeIdx]);
            
            float2 sampleUV = (texelPos + 0.5f) / float2(upperTexWidth, upperTexHeight);
            float4 sample = upperRadianceTexture.sample(samplerLinear, sampleUV);
            
            probeRadiance += sample * dirBilinearWeights[dirIdx];
        }
        
        accumulatedRadiance += probeRadiance * probeWeight;
    }

    return accumulatedRadiance;
}

// Manually uncomment anything related to debugging if you want to see the probe data
kernel void raytracingKernel(texture2d<float, access::write>    radianceTexture         [[texture(TextureIndexRadiance)]],
                             texture2d<float, access::sample>   upperRadianceTexture    [[texture(TextureIndexRadianceUpper)]],
                    constant FrameData&                         frameData               [[buffer(BufferIndexFrameData)]],
                    constant CascadeData&                       cascadeData             [[buffer(BufferIndexCascadeData)]],
                             primitive_acceleration_structure   accelerationStructure   [[buffer(BufferIndexAccelerationStructure)]],
                const device TriangleResources::TriangleData*   resources               [[buffer(BufferIndexResources)]],
//                      device Probe*                             probeData               [[buffer(BufferIndexProbeData)]],
//                      device ProbeRay*                          rayData                 [[buffer(BufferIndexProbeRayData)]],
                             texture2d<float, access::sample>   depthTexture            [[texture(TextureIndexDepthTexture)]],
                             uint                               tid                     [[thread_position_in_grid]]) {
    const uint probeSpacing = cascadeData.probeSpacing;
    uint cascadeLevel = cascadeData.cascadeLevel;
    float intervalLength = cascadeData.intervalLength;

    // Compute probe grid
    uint tileSize = probeSpacing * (1 << cascadeLevel);
    uint width = frameData.framebuffer_width;
    uint height = frameData.framebuffer_height;
    uint probeGridSizeX = (width + tileSize - 1) / tileSize;
    uint probeGridSizeY = (height + tileSize - 1) / tileSize;

    uint raysPerDim = (1 << (cascadeLevel + 2)); // 4, 8, 16, ...
    uint totalProbes = probeGridSizeX * probeGridSizeY;
    uint totalDirections = raysPerDim * raysPerDim; // 16, 64, 256
    uint totalThreads = totalProbes * totalDirections;

    if (tid >= totalThreads) {
        return;
    }

    // DIRECTION-FIRST
    uint directionIndex = tid / totalProbes;  // Which direction is this thread processing
    uint probeIndex = tid % totalProbes;      // Which probe within that direction

    // Convert direction index to 2D coordinates in the octahedron map
    uint dirX = directionIndex % raysPerDim;
    uint dirY = directionIndex / raysPerDim;
    
    // Center of pixel
    float2 dirUV = float2((dirX + 0.5f) / float(raysPerDim),
                          (dirY + 0.5f) / float(raysPerDim));

    float3 rayDir = octDecode(dirUV);
    
    // Calculate probe position in the 2D grid
    uint probeX = probeIndex % probeGridSizeX;
    uint probeY = probeIndex / probeGridSizeX;
    
    // Map probe to screen UV center
    float2 probeUV = (float2(probeX, probeY) + 0.5f) / float2(probeGridSizeX, probeGridSizeY);
    float2 probeNDC = probeUV * 2.0f - 1.0f;
    probeNDC.y = -probeNDC.y; // Flip Y axis for Metal API
    float probeDepth = depthTexture.sample(depthSampler, probeUV).x;
    float3 worldPos = reconstructWorldPositionFromLinearDepth(probeNDC, probeDepth, frameData);
    
    // Store probe position for debugging
//    if (rayDir.x == 0 && rayDir.y == 0 && rayDir.z == 1.0) {
//        // Only store once per probe (for the ray pointing straight ahead)
//        probeData[probeIndex].position = float4(worldPos, 0.0f);
//    }
    
    // Calculate cascade ranges
    const float baseCascadeRange = 0.4f; // Magic number determined by trial and error
    const float cascadeRangeMultiplier = 4.0f; // The branching factor
    float cascadeStartRange = (cascadeLevel == 0) ? 0.0f : (baseCascadeRange * pow(cascadeRangeMultiplier, float(cascadeLevel - 1)));
    float cascadeEndRange = baseCascadeRange * pow(cascadeRangeMultiplier, float(cascadeLevel));
    float intervalStart = cascadeStartRange * intervalLength;
    float intervalEnd = cascadeEndRange * intervalLength;
    
    // Setup ray for tracing
    ray ray;
    ray.origin = worldPos;
    ray.direction = rayDir;
    ray.min_distance = intervalStart;
    ray.max_distance = intervalEnd;

    // Store ray interval for debugging
//    uint rayDataIndex = probeIndex * totalDirections + directionIndex;
//    float3 startPoint = worldPos + rayDir * intervalStart;
//    float3 endPoint = worldPos + rayDir * intervalEnd;
//    rayData[rayDataIndex].intervalStart = float4(startPoint, 1.0);
//    rayData[rayDataIndex].intervalEnd = float4(endPoint, 1.0);
    
    // Perform ray-intersection test
    intersector<triangle_data> intersector;
    intersection_result<triangle_data> result = intersector.intersect(ray, accelerationStructure);
    
    // Direct radiance from current cascade
    bool sampleSunOrSky = cascadeData.enableSky || cascadeData.enableSun;
    float4 radiance = float4(0.0);
    float occlusion;
    
    if (result.type != intersection_type::none) {
        // Hit something - check if emissive
        unsigned int primitiveIndex = result.primitive_id;
        const device TriangleResources::TriangleData& triangle = resources[primitiveIndex];
        // If -1.0, it is emissive
        radiance = (triangle.colors[0].a == -1.0f) ? float4(triangle.colors[0].rgb, 1.0) : float4(0.0, 0.0, 0.0, 1.0);
        occlusion = 0.0;
        
        // Set ray color for debugging
//        if (triangle.colors[0].a == -1.0f)
//            rayData[rayDataIndex].color = float4(1.0, 0.0, 0.0, 1.0); // Red for emissive
//        else
//            rayData[rayDataIndex].color = float4(0.0, 0.0, 1.0, 1.0); // Blue for non-emissive
    } else {
        occlusion = 1.0f;
        // No intersection - Apply sky for higher cascades
        if ((cascadeLevel == cascadeData.maxCascade) && sampleSunOrSky) {
            radiance = skyAndSun(rayDir, frameData, cascadeData);
//            rayData[rayDataIndex].color = float4(0.0, 1.0, 1.0, 1.0); // Cyan for sky
        } else {
            radiance = float4(0.0, 0.0, 0.0, 1.0);
//            rayData[rayDataIndex].color = float4(0.4, 0.4, 0.4, 1.0); // Gray for misses
        }
    }

    // Merge with upper cascade
    float4 upperRadiance = float4(0.0);
    if (cascadeLevel < cascadeData.maxCascade) {
        upperRadiance = mergeUpperCascade(upperRadianceTexture,
                                          depthTexture,
                                          probeUV,
                                          rayDir,
                                          probeDepth,
                                          worldPos,
                                          cascadeData,
                                          frameData);

        if (result.type == intersection_type::none) {
            // If no hit in current cascade, use upper cascade
            radiance = upperRadiance;
        } else {
            // There was a hit, blend with upper cascade
            radiance.rgb += upperRadiance.rgb * occlusion;
            radiance.a *= upperRadiance.a;
        }
    }
    
    // Calculate output texture coordinates using direction-first layout
    // For each direction (dirX, dirY), we allocate a complete grid of probes
    // Each direction's grid starts at (dirX * probeGridSizeX, dirY * probeGridSizeY)
    // And within that grid, we place the probe at (probeX, probeY)
    uint texX = dirX * probeGridSizeX + probeX;
    uint texY = dirY * probeGridSizeY + probeY;
    
    radianceTexture.write(radiance, uint2(texX, texY));
}
