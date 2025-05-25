#define METAL
#include <metal_stdlib>
using namespace metal;

#include "shaderTypes.hpp"
#include "shaderCommon.hpp"
#include "common.hpp"

struct VertexOut {
    float4 position [[position]];
    float2 texCoords;
#if USE_EYE_DEPTH
    float3 eye_position;
#endif
};

vertex VertexOut final_gather_vertex(uint       vertexID    [[vertex_id]],
                            constant FrameData& frameData   [[buffer(BufferIndexFrameData)]]) {
    VertexOut out;
    
    float2 position = float2((vertexID << 1) & 2, vertexID & 2);
    out.position = float4(position * 2.0f - 1.0f, 0.0f, 1.0f);
    out.texCoords = position;

#if USE_EYE_DEPTH
    float4 unprojected_eye_coord = frameData.projection_matrix_inverse * out.position;
    out.eye_position = unprojected_eye_coord.xyz / unprojected_eye_coord.w;
#endif
    
    return out;
}

half3 gammaCorrect(half3 linear) {
    return pow(linear, 1.0f / 2.2f);
}

half3 acesTonemap(half3 color) {
    half3 a = 2.51h;
    half3 b = 0.03h;
    half3 c = 2.43h;
    half3 d = 0.59h;
    half3 e = 0.14h;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

half3 postProcessColor(half3 color) {
    color = acesTonemap(color);
    color = gammaCorrect(color);
    
    return color;
}

half3 renderSky(float3 rayDir, FrameData frameData) {
    float sunIntensity = frameData.sun_specular_intensity;
    
    const float3 skyZenithColor = float3(0.1, 0.2, 0.6);      // Deeper blue at zenith
    const float3 skyMidColor = float3(0.4, 0.6, 0.9);         // Mid sky blue
    const float3 skyHorizonColor = float3(0.8, 0.9, 0.95);    // Almost white at horizon
    
    float height = 0.5;
    float3 skyGradient;
    
    if (height < 0.3) {
        // Lower portion: horizon color to mid
        float t = height / 0.3;
        skyGradient = mix(skyHorizonColor, skyMidColor, smoothstep(0.0, 1.0, t));
    } else if (height < 0.5) {
        // Middle portion: solid mid color
        skyGradient = skyMidColor;
    } else {
        // Upper portion: mid to zenith
        float t = (height - 0.5) / 0.5;
        skyGradient = mix(skyMidColor, skyZenithColor, smoothstep(0.0, 1.0, t));
    }
    
    float skyIntensity = sunIntensity * 0.7f;
    float3 finalSkyColor = skyGradient * skyIntensity;
    
    return half3(finalSkyColor);
}

fragment half4 final_gather_fragment(VertexOut in [[stage_in]],
                         constant    FrameData&          frameData       [[buffer(BufferIndexFrameData)]],
                                     texture2d<float>    radianceTexture [[texture(TextureIndexRadiance)]],
                                     texture2d<half>     albedoTexture   [[texture(TextureIndexBaseColor)]],
                                     texture2d<half>     normalTexture   [[texture(TextureIndexNormal)]],
                                     texture2d<float>    depthTexture    [[texture(TextureIndexDepthTexture)]],
                         constant    bool&               doBilinear      [[buffer(3)]],
                         constant    bool&               drawSky         [[buffer(4)]]) {
    float2 texCoords = float2(in.texCoords.x, 1.0 - in.texCoords.y);

    // Sample G-Buffer textures
    half4 albedoSpecular = half4(albedoTexture.sample(samplerLinear, texCoords));
    half3 albedo = albedoSpecular.rgb;
    half4 normalRaw = half4(normalTexture.sample(samplerLinear, texCoords));
    half3 normal = normalize(normalRaw.xyz);
    float currentDepth = depthTexture.sample(samplerLinear, texCoords).x;
    
    // If pixel is empty, draw the sky
    if (length(albedoSpecular.rgb) == 0.0) {
        if (drawSky) {
            float4 clipPos = float4(in.texCoords * 2.0 - 1.0, 0.0, 1.0);
            clipPos.y = -clipPos.y;
            float4 viewDir = frameData.projection_matrix_inverse * clipPos;
            float3 worldDir = normalize((frameData.view_matrix_inverse * viewDir).xyz);
            
            half3 skyColor = renderSky(worldDir, frameData);
            return half4(postProcessColor(skyColor), 1.0h);
        }
        
        return half4(0.0, 0.0, 0.0, 1.0);
    }
    
    bool isEmissive = (normalRaw.a == -1);
    
    if (!doBilinear) {
        float4 radiance = radianceTexture.sample(samplerLinear, texCoords);
        return half4(half3(radiance.rgb), 1.0);
    }
    
    // Calculate probe grid coordinates
    float2 probeGridSize = float2(frameData.framebuffer_width * 0.25, frameData.framebuffer_height * 0.25);
    float2 probeCoord = texCoords * probeGridSize - 0.5f;
    int2 probeBase = int2(floor(probeCoord));
    float2 probeFrac = fract(probeCoord);
    probeBase = clamp(probeBase, int2(0), int2(probeGridSize) - int2(2));

    int2 probeOffsets[4] = {int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1)};
    float4 probeDepths;
    int2 probeCoords[4];
    float2 probeUVs[4];
    
    for (int i = 0; i < 4; i++) {
        probeCoords[i] = clamp(int2(probeBase) + probeOffsets[i], int2(0.0), int2(probeGridSize-1));
        probeUVs[i] = (float2(probeCoords[i]) + 0.5f) / probeGridSize;
        probeDepths[i] = depthTexture.sample(samplerLinear, probeUVs[i]).x;
    }

    // Calculate bilinear weights with depth-aware filtering
    float4 bilinearWeights = float4((1.0f - probeFrac.x)    * (1.0f - probeFrac.y),
                                   probeFrac.x              * (1.0f - probeFrac.y),
                                   (1.0f - probeFrac.x)     * probeFrac.y,
                                   probeFrac.x              * probeFrac.y);

    // Apply bilateral filtering for depth discontinuities
    float mind = min(min(probeDepths.x, probeDepths.y), min(probeDepths.z, probeDepths.w));
    float maxd = max(max(probeDepths.x, probeDepths.y), max(probeDepths.z, probeDepths.w));
    float diffd = maxd - mind;
    float avg = dot(probeDepths, float4(0.25f));
    bool d_edge = (diffd / avg) > 0.05; // Depth discontinuity threshold

    float4 w = bilinearWeights;
    if (d_edge) {
        float4 dd = abs(probeDepths - float4(currentDepth));
        w *= float4(1.0f) / (dd + float4(0.0001f));
    }
    
    float wsum = w.x + w.y + w.z + w.w;
    w /= wsum;

    // Direction-first layout sampling parameters
    const uint probeTileSize = 4; // Cascade 0 octahedrons are always 4x4
    uint probeGridSizeX = uint(probeGridSize.x);
    uint probeGridSizeY = uint(probeGridSize.y);

    float4 probeRadiance[4];
    
    // Sample radiance for each probe using direction-first layout
    for (int probeIdx = 0; probeIdx < 4; probeIdx++) {
        float4 radianceSum = float4(0.0);
        float totalWeight = 0.0;
        
        for (uint directionIndex = 0; directionIndex < (probeTileSize * probeTileSize); directionIndex++) {
            uint dirX = directionIndex % probeTileSize;
            uint dirY = directionIndex / probeTileSize;
            
            float2 dirUV = float2((dirX + 0.5f) / probeTileSize, (dirY + 0.5f) / probeTileSize);
            float3 direction = octDecode(dirUV);
            float cosTheta = max(0.0, dot(float3(normal), direction));
            
            // Calculate texture coordinates for direction-first layout
            // Each direction (dirX, dirY) has its own complete probe grid
            uint2 texCoord = uint2(dirX * probeGridSizeX + probeCoords[probeIdx].x,
                                   dirY * probeGridSizeY + probeCoords[probeIdx].y);
            
            float2 sampleUV = (float2(texCoord) + 0.5f) / float2(probeTileSize * probeGridSizeX,
                                                                 probeTileSize * probeGridSizeY);
            
            if (isEmissive) {
                float3 emissiveContribution = float3(albedo) * cosTheta;
                radianceSum += float4(emissiveContribution, 1.0) * cosTheta;
            } else {
                float4 sample = radianceTexture.sample(samplerLinear, sampleUV);
                radianceSum += sample * cosTheta;
            }
            
            totalWeight += cosTheta;
        }
        
        probeRadiance[probeIdx] = (totalWeight > 0.0001f) ? (radianceSum / totalWeight) : float4(0.0);
    }

    float4 finalRadiance = probeRadiance[0] * w.x +
                           probeRadiance[1] * w.y +
                           probeRadiance[2] * w.z +
                           probeRadiance[3] * w.w;

    half3 finalColor = albedo * half3(finalRadiance.rgb);
    
    return half4(postProcessColor(finalColor), 1.0h);
}
