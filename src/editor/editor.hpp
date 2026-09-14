#pragma once

#include "pch.hpp"

#include <Metal/Metal.hpp>
#include <GLFW/glfw3.h>
#include <simd/simd.h>
#include "../../external/imgui/imgui.h"

class Editor {
public:

    struct DebugWindowOptions {
        bool enableDebugFeature = false;
        bool sky = false;
        bool sun = false;
        int debugCascadeLevel = -1;
        float intervalLength = 0.6f;
        simd::float3 cameraPosition = simd::float3{7.0f, 5.0f, 0.0f};
    } debug;

    Editor(GLFWwindow* window, MTL::Device* device);
    ~Editor();

    void beginFrame(MTL::RenderPassDescriptor* passDescriptor);
    void endFrame(MTL::CommandBuffer* commandBuffer, MTL::RenderCommandEncoder* encoder);
    void cleanup();
    
private:
    GLFWwindow* window;
    MTL::Device* device;

    void createDockSpace();
    void debugWindow();
};
