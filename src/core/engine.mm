#include "engine.hpp"

Engine::Engine()
: camera(simd::float3{7.0f, 5.0f, 0.0f}, NEAR_PLANE, FAR_PLANE)
, lastFrame(0.0f)
, frameNumber(0)
, currentFrameIndex(0) {
	inFlightSemaphore = dispatch_semaphore_create(MaxFramesInFlight);

    for (int i = 0; i < MaxFramesInFlight; i++) {
        frameSemaphores[i] = dispatch_semaphore_create(1);
    }
}

void Engine::init() {
    initDevice();
    initWindow();

    resourceManager = std::make_unique<ResourceManager>(metalDevice);
    editor = std::make_unique<Editor>(glfwWindow, metalDevice);
    debug = std::make_unique<Debug>(metalDevice);
    rayTracingManager = std::make_unique<RayTracingManager>(metalDevice, resourceManager.get());

    createCommandQueue();
	loadScene();
    createDefaultLibrary();
    createBuffers();
    renderPipelines.initialize(metalDevice, metalDefaultLibrary);
    defaultVertexDescriptor = createDefaultVertexDescriptor();
    createRenderPipelines();

    renderPassManager = std::make_unique<RenderPassManager>(
        metalDevice, 
        resourceManager.get(), 
        &renderPipelines, 
        rayTracingManager.get(), 
        debug.get(), 
        editor.get()
    );
    
	createViewRenderPassDescriptor();
    rayTracingManager->setupAccelerationStructures(meshes);
    rayTracingManager->setupTriangleResources(meshes);
}

void Engine::run() {
    while (!glfwWindowShouldClose(glfwWindow)) {
        float currentFrame = glfwGetTime();
        float deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        
        camera.processKeyboardInput(glfwWindow, deltaTime);
        editor->debug.cameraPosition = camera.position;
        
        @autoreleasepool {
            metalDrawable = (__bridge CA::MetalDrawable*)[metalLayer nextDrawable];
            draw();
        }
        
        glfwPollEvents();
    }
}

void Engine::cleanup() {
    glfwTerminate();
    
    // Clean up mesh objects
    for (auto& mesh : meshes) {
        delete mesh;
    }
    
    // Release frame-specific buffers that are not managed by ResourceManager
    for (int frame = 0; frame < MaxFramesInFlight; frame++) {
        // Will be moving these to ResourceManager eventually, but for now:
        for (int cascade = 0; cascade < MAX_CASCADE_LEVEL; cascade++) {
            if (cascadeDataBuffer[frame][cascade]) {
                cascadeDataBuffer[frame][cascade]->release();
            }
            
            if (createDebugData) {
                if (probePosBuffer[frame][cascade]) {
                    probePosBuffer[frame][cascade]->release();
                }
                if (rayBuffer[frame][cascade]) {
                    rayBuffer[frame][cascade]->release();
                }
            }
        }
    }
    
    // Release descriptors which aren't managed by ResourceManager
    if (viewRenderPassDescriptor) viewRenderPassDescriptor->release();
    if (finalGatherDescriptor) finalGatherDescriptor->release();
    if (forwardDescriptor) forwardDescriptor->release();
    if (depthPrepassDescriptor) depthPrepassDescriptor->release();
    
    // Release other Metal objects
    if (defaultVertexDescriptor) defaultVertexDescriptor->release();
    if (metalCommandQueue) metalCommandQueue->release();
    
    // Use ResourceManager to release all resources it manages
    if (resourceManager) {
        resourceManager->releaseAllResources();
    }
    
    // Finally release the device
    if (metalDevice) {
        metalDevice->release();
    }
}

void Engine::initDevice() {
    metalDevice = MTL::CreateSystemDefaultDevice();
}

void Engine::frameBufferSizeCallback(GLFWwindow *window, int width, int height) {
    Engine* engine = (Engine*)glfwGetWindowUserPointer(window);
    engine->resizeFrameBuffer(width, height);
}

void Engine::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    Engine* engine = (Engine*)glfwGetWindowUserPointer(window);
    engine->camera.processMouseButton(window, button, action);
}

void Engine::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    Engine* engine = (Engine*)glfwGetWindowUserPointer(window);
    engine->camera.processMouseMovement(xpos, ypos);
}

void Engine::resizeFrameBuffer(int width, int height) {
    metalLayer.drawableSize = CGSizeMake(width, height);
    
    // Release render pass descriptors
    if (viewRenderPassDescriptor)   viewRenderPassDescriptor->release();
    if (finalGatherDescriptor)      finalGatherDescriptor->release();
    if (forwardDescriptor)          forwardDescriptor->release();
    if (depthPrepassDescriptor)     depthPrepassDescriptor->release();
    
    viewRenderPassDescriptor = nil;
    finalGatherDescriptor = nil;
    forwardDescriptor = nil;
    depthPrepassDescriptor = nil;

    // Recreate view render pass descriptor, which will recreate all textures
    createViewRenderPassDescriptor();
    
    // Get new drawable and update descriptors
    metalDrawable = (__bridge CA::MetalDrawable*)[metalLayer nextDrawable];
    updateRenderPassDescriptor();
}

void Engine::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindow = glfwCreateWindow(1280, 1280, "RC-SPWI", NULL, NULL);
    if (!glfwWindow) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    int width, height;
    glfwGetFramebufferSize(glfwWindow, &width, &height);

    metalWindow = glfwGetCocoaWindow(glfwWindow);
    metalLayer = [CAMetalLayer layer];
    metalLayer.device = (__bridge id<MTLDevice>)metalDevice;
    metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    metalLayer.drawableSize = CGSizeMake(width, height);
    metalWindow.contentView.layer = metalLayer;
    metalWindow.contentView.wantsLayer = YES;
    metalLayer.framebufferOnly = false;

    glfwSetWindowUserPointer(glfwWindow, this);
    glfwSetFramebufferSizeCallback(glfwWindow, frameBufferSizeCallback);
    glfwSetMouseButtonCallback(glfwWindow, mouseButtonCallback);
    glfwSetCursorPosCallback(glfwWindow, cursorPosCallback);
    lastFrame = glfwGetTime();
    
    metalDrawable = (__bridge CA::MetalDrawable*)[metalLayer nextDrawable];
}

MTL::CommandBuffer* Engine::beginFrame(bool isPaused) {
    // Wait on the semaphore for the current frame
    dispatch_semaphore_wait(frameSemaphores[currentFrameIndex], DISPATCH_TIME_FOREVER);

	MTL::CommandBuffer* commandBuffer = metalCommandQueue->commandBuffer();
	
	MTL::CommandBufferHandler handler = [this](MTL::CommandBuffer*) {
		// Signal the semaphore for this frame when GPU work is complete
		dispatch_semaphore_signal(frameSemaphores[currentFrameIndex]);
	};
	commandBuffer->addCompletedHandler(handler);
    
    updateWorldState(isPaused);
	
	return commandBuffer;
}

void Engine::endFrame(MTL::CommandBuffer* commandBuffer) {
    if(commandBuffer) {
        commandBuffer->presentDrawable(metalDrawable);
        commandBuffer->commit();
        
        // Draw the debug spheres and lines once after 100 frames so you can inspect them
        // Remove the if statement for real-time updates. Slow as fuck
        if (frameNumber == 100 && createDebugData) {
            createSphereGrid();
            createDebugLines();
        }
        
        // Move to next frame
        currentFrameIndex = (currentFrameIndex + 1) % MaxFramesInFlight;
    }
}

void Engine::loadSceneFromJSON(const std::string& jsonFilePath) {
    if (!defaultVertexDescriptor) {
        defaultVertexDescriptor = createDefaultVertexDescriptor();
    }
    
    SceneParser parser(metalDevice, defaultVertexDescriptor);
    
    std::vector<Mesh*> loadedMeshes = parser.loadScene(jsonFilePath);
    
    meshes.insert(meshes.end(), loadedMeshes.begin(), loadedMeshes.end());
}

void Engine::loadScene() {
    loadSceneFromJSON(std::string(SCENES_PATH) + "/sponza.json");
}

MTL::VertexDescriptor* Engine::createDefaultVertexDescriptor() {
    MTL::VertexDescriptor* vertexDescriptor = MTL::VertexDescriptor::alloc()->init();
    
    // Position attribute
    vertexDescriptor->attributes()->object(VertexAttributePosition)->setFormat(MTL::VertexFormatFloat4);
    vertexDescriptor->attributes()->object(VertexAttributePosition)->setOffset(offsetof(Vertex, position));
    vertexDescriptor->attributes()->object(VertexAttributePosition)->setBufferIndex(0);
    
    // Normal attribute
    vertexDescriptor->attributes()->object(VertexAttributeNormal)->setFormat(MTL::VertexFormatFloat4);
    vertexDescriptor->attributes()->object(VertexAttributeNormal)->setOffset(offsetof(Vertex, normal));
    vertexDescriptor->attributes()->object(VertexAttributeNormal)->setBufferIndex(0);
    
    // IMPORTANT: Always include all attributes for pipeline creation, even for non-textured meshes
    // The function constants will handle their usage in the shader
    
    // Texture coordinate attribute
    vertexDescriptor->attributes()->object(VertexAttributeTexcoord)->setFormat(MTL::VertexFormatFloat2);
    vertexDescriptor->attributes()->object(VertexAttributeTexcoord)->setOffset(offsetof(Vertex, textureCoordinate));
    vertexDescriptor->attributes()->object(VertexAttributeTexcoord)->setBufferIndex(0);
    
    // Tangent attribute
    vertexDescriptor->attributes()->object(VertexAttributeTangent)->setFormat(MTL::VertexFormatFloat4);
    vertexDescriptor->attributes()->object(VertexAttributeTangent)->setOffset(offsetof(Vertex, tangent));
    vertexDescriptor->attributes()->object(VertexAttributeTangent)->setBufferIndex(0);
    
    // Bitangent attribute
    vertexDescriptor->attributes()->object(VertexAttributeBitangent)->setFormat(MTL::VertexFormatFloat4);
    vertexDescriptor->attributes()->object(VertexAttributeBitangent)->setOffset(offsetof(Vertex, bitangent));
    vertexDescriptor->attributes()->object(VertexAttributeBitangent)->setBufferIndex(0);
    
    // DiffuseTextureIndex
    vertexDescriptor->attributes()->object(VertexAttributeDiffuseIndex)->setFormat(MTL::VertexFormatInt);
    vertexDescriptor->attributes()->object(VertexAttributeDiffuseIndex)->setOffset(offsetof(Vertex, diffuseTextureIndex));
    vertexDescriptor->attributes()->object(VertexAttributeDiffuseIndex)->setBufferIndex(0);
    
    // NormalTextureIndex
    vertexDescriptor->attributes()->object(VertexAttributeNormalIndex)->setFormat(MTL::VertexFormatInt);
    vertexDescriptor->attributes()->object(VertexAttributeNormalIndex)->setOffset(offsetof(Vertex, normalTextureIndex));
    vertexDescriptor->attributes()->object(VertexAttributeNormalIndex)->setBufferIndex(0);
    
    // Set layout
    vertexDescriptor->layouts()->object(0)->setStride(sizeof(Vertex));
    vertexDescriptor->layouts()->object(0)->setStepRate(1);
    vertexDescriptor->layouts()->object(0)->setStepFunction(MTL::VertexStepFunctionPerVertex);
    
    return vertexDescriptor;
}

void Engine::createDefaultLibrary() {
    // Create an NSString from the metallib path
    NS::String* libraryPath = NS::String::string(
        SHADER_METALLIB,
        NS::UTF8StringEncoding
    );
    
    NS::Error* error = nullptr;

    printf("Selected Device: %s\n", metalDevice->name()->utf8String());
    
    metalDefaultLibrary = metalDevice->newLibrary(libraryPath, &error);
    
    if (!metalDefaultLibrary) {
        std::cerr << "Failed to load metal library at path: " << SHADER_METALLIB;
        if (error) {
            std::cerr << "\nError: " << error->localizedDescription()->utf8String();
        }
        std::exit(-1);
    }
}

void Engine::createBuffers() {
    cascadeDataBuffer.resize(MaxFramesInFlight);
    probePosBuffer.resize(MaxFramesInFlight);
    frameDataBuffers.resize(MaxFramesInFlight);
    rayBuffer.resize(MaxFramesInFlight);
    
    for (int frame = 0; frame < MaxFramesInFlight; frame++) {
        std::string labelStr = "FrameData: " + std::to_string(frame);
        frameDataBuffers[frame] = resourceManager->createBuffer(
            sizeof(FrameData), 
            nullptr, 
            MTL::ResourceStorageModeShared, 
            labelStr.c_str()
        );
        
        cascadeDataBuffer[frame].resize(MAX_CASCADE_LEVEL);
        probePosBuffer[frame].resize(MAX_CASCADE_LEVEL);
        rayBuffer[frame].resize(MAX_CASCADE_LEVEL);
        
        for (int cascade = 0; cascade < MAX_CASCADE_LEVEL; cascade++) {
            labelStr = "Frame: " + std::to_string(frame) + "|CascadeData: " + std::to_string(cascade);
            
            cascadeDataBuffer[frame][cascade] = resourceManager->createBuffer(
                sizeof(CascadeData), 
                nullptr, 
                MTL::ResourceStorageModeShared, 
                labelStr.c_str()
            );

            if (createDebugData) {
                debugProbeCount = floor(metalLayer.drawableSize.width / (PROBE_SPACING * 1 << cascade)) * 
                                  floor(metalLayer.drawableSize.height / (PROBE_SPACING * 1 << cascade));
                size_t probeBufferSize = debugProbeCount * sizeof(Probe);
                rayCount = debugProbeCount * BASE_RAY * (1 << (2 * cascade));
                size_t rayBufferSize = rayCount * sizeof(ProbeRay);
                
                labelStr = "Frame: " + std::to_string(frame) + "|CascadeProbes: " + std::to_string(cascade);
                probePosBuffer[frame][cascade] = resourceManager->createBuffer(
                    probeBufferSize, 
                    nullptr, 
                    MTL::ResourceStorageModeShared, 
                   labelStr.c_str()
                );
                
                labelStr = "Frame: " + std::to_string(frame) + "|CascadeRays: " + std::to_string(cascade);
                rayBuffer[frame][cascade] = resourceManager->createBuffer(
                    rayBufferSize, 
                    nullptr, 
                    MTL::ResourceStorageModeShared, 
                  labelStr.c_str()
                );
            }
        }
    }
}

void printMatrix(const matrix_float4x4& matrix) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << matrix.columns[j][i] << " ";
        }
        std::cout << std::endl;
    }
}

void Engine::updateWorldState(bool isPaused) {
	if (!isPaused) {
		frameNumber++;
	}

	FrameData *frameData = (FrameData *)(frameDataBuffers[currentFrameIndex]->contents());

	float aspectRatio = metalDrawable->layer()->drawableSize().width / metalDrawable->layer()->drawableSize().height;
	
	camera.setProjectionMatrix(45, aspectRatio, NEAR_PLANE, FAR_PLANE);
	frameData->projection_matrix = camera.getProjectionMatrix();
	frameData->projection_matrix_inverse = matrix_invert(frameData->projection_matrix);
	frameData->view_matrix = camera.getViewMatrix();
    frameData->view_matrix_inverse = camera.getInverseViewMatrix();
    
    frameData->cameraUp         = float4{camera.up.x,       camera.up.y,        camera.up.z, 1.0f};
    frameData->cameraRight      = float4{camera.right.x,    camera.right.y,     camera.right.z, 1.0f};
    frameData->cameraForward    = float4{camera.front.x,    camera.front.y,     camera.front.z, 1.0f};
    frameData->cameraPosition   = float4{camera.position.x, camera.position.y,  camera.position.z, 1.0f};

	// Set screen dimensions
	frameData->framebuffer_width = (uint)metalLayer.drawableSize.width;
	frameData->framebuffer_height = (uint)metalLayer.drawableSize.height;
    frameData->near_plane = NEAR_PLANE;
    frameData->far_plane = FAR_PLANE;

	// Define the sun color
	frameData->sun_color = simd_make_float4(0.95, 0.95, 0.9, 1.0);
	frameData->sun_specular_intensity = 0.7;

	// Calculate the sun's X position oscillating over time
	float oscillationSpeed = 0.02f;
	float oscillationAmplitude = 9.0f;
	float sunZ = sin(frameNumber * oscillationSpeed) * oscillationAmplitude;

	float sunY = 10.0f;
	float sunX = 0.0f;

	// Sun world position
	float4 sunWorldPosition = {sunX, sunY, sunZ, 1.0};
	float4 sunWorldDirection = -sunWorldPosition;

	// Update the sun direction in view space
	frameData->sun_eye_direction = sunWorldDirection;

	float4 directionalLightUpVector = {0.0, 1.0, 0.0, 0.0};
	// Update scene matrices
	frameData->scene_model_matrix = matrix4x4_translation(0.0f, 0.0f, 0.0f); // Sponza at origin
	frameData->scene_modelview_matrix = frameData->view_matrix * frameData->scene_model_matrix;
	frameData->scene_normal_matrix = matrix3x3_upper_left(frameData->scene_model_matrix);
}

void Engine::createCommandQueue() {
    metalCommandQueue = metalDevice->newCommandQueue();
}

void Engine::createRenderPipelines() {
    NS::Error* error;
	
	albedoSpecularGBufferFormat = MTL::PixelFormatRGBA8Unorm_sRGB;
	normalMapGBufferFormat 	    = MTL::PixelFormatRGBA8Snorm;
	depthGBufferFormat			= MTL::PixelFormatR32Float;

    #pragma mark Deferred render pipeline setup
    {
		{
			RenderPipelineConfig gbufferConfig{
                .label = "G-buffer Creation",
                .vertexFunctionName = "gbuffer_vertex",
                .fragmentFunctionName = "gbuffer_fragment",
                .vertexDescriptor = defaultVertexDescriptor
            };
            gbufferConfig.colorAttachments = {
                {RenderTargetAlbedo, albedoSpecularGBufferFormat},
                {RenderTargetNormal, normalMapGBufferFormat},
                {RenderTargetDepth, depthGBufferFormat}
            };
            
            // Create the function constant object
            static NS::String* hasTexturesID = NS::String::string("hasTextures", NS::ASCIIStringEncoding);
            MTL::FunctionConstantValues* hasTexturesTrue = MTL::FunctionConstantValues::alloc()->init();
            bool trueValue = true;
            hasTexturesTrue->setConstantValue(&trueValue, MTL::DataTypeBool, hasTexturesID);

            MTL::FunctionConstantValues* hasTexturesFalse = MTL::FunctionConstantValues::alloc()->init();
            bool falseValue = false;
            hasTexturesFalse->setConstantValue(&falseValue, MTL::DataTypeBool, hasTexturesID);

            // Create pipelines for both cases
            RenderPipelineConfig gbufferTexturedConfig = gbufferConfig;
            gbufferTexturedConfig.functionConstants = hasTexturesTrue;
            renderPipelines.createRenderPipeline(RenderPipelineType::GBufferTextured, gbufferTexturedConfig);

            RenderPipelineConfig gbufferNonTexturedConfig = gbufferConfig;
            gbufferNonTexturedConfig.functionConstants = hasTexturesFalse;
            renderPipelines.createRenderPipeline(RenderPipelineType::GBufferNonTextured, gbufferNonTexturedConfig);
            
            RenderPipelineConfig depthPrepassConfig{
                .label = "Depth Prepass",
                .vertexFunctionName = "depth_prepass_vertex",
                .fragmentFunctionName = "depth_prepass_fragment",
                .colorPixelFormat = MTL::PixelFormatR32Float,
                .depthPixelFormat = MTL::PixelFormatDepth32Float_Stencil8,
                .stencilPixelFormat = MTL::PixelFormatDepth32Float_Stencil8,
                .vertexDescriptor = defaultVertexDescriptor,
                .functionConstants = hasTexturesFalse
            };
            renderPipelines.createRenderPipeline(RenderPipelineType::DepthPrepass, depthPrepassConfig);
            
            // Create depth stencil state for depth prepass
            DepthStencilConfig depthPrepassDepthConfig{
                .label = "Depth Prepass Depth State",
                .depthCompareFunction = MTL::CompareFunctionLess,
                .depthWriteEnabled = true
            };
            renderPipelines.createDepthStencilState(DepthStencilType::DepthPrepass, depthPrepassDepthConfig);
		}
		
		#pragma mark GBuffer depth state setup
		{
		#if LIGHT_STENCIL_CULLING
			StencilConfig gbufferStencil{
                .stencilCompareFunction = MTL::CompareFunctionAlways,
                .stencilFailureOperation = MTL::StencilOperationKeep,
                .depthFailureOperation = MTL::StencilOperationKeep,
                .depthStencilPassOperation = MTL::StencilOperationReplace,
                .readMask = 0x0,
                .writeMask = 0xFF
            };
		#else
			StencilConfig gbufferStencil{};
		#endif
			DepthStencilConfig gbufferDepthConfig{
                .label = "G-buffer Creation",
                .depthCompareFunction = MTL::CompareFunctionLess,
                .depthWriteEnabled = true,
                .frontStencil = gbufferStencil,
                .backStencil = gbufferStencil
            };
            renderPipelines.createDepthStencilState(DepthStencilType::GBuffer, gbufferDepthConfig);
		}
		
		// Setup render state to apply the radiance light in final pass
		{
            #pragma mark Final gathering render pipeline setup
            {
                RenderPipelineConfig finalGatherConfig{
                    .label = "Final Gathering",
                    .vertexFunctionName = "final_gather_vertex",
                    .fragmentFunctionName = "final_gather_fragment",
                    .colorPixelFormat = metalDrawable->texture()->pixelFormat(),
                    .depthPixelFormat = MTL::PixelFormatInvalid,
                    .stencilPixelFormat = MTL::PixelFormatInvalid,
                    .vertexDescriptor = nullptr
                };
                renderPipelines.createRenderPipeline(RenderPipelineType::FinalGather, finalGatherConfig);
            }
		}
    }
    
    #pragma mark Ray tracing pipeline state
    {
        ComputePipelineConfig raytracingConfig{
            .label = "Raytracing Pipeline",
            .computeFunctionName = "raytracingKernel"
        };
        renderPipelines.createComputePipeline(ComputePipelineType::Raytracing, raytracingConfig);
    }
    
    #pragma mark Forward Debug pipeline state
    {
        RenderPipelineConfig debugConfig{
            .label = "Forward Debug Pipeline",
            .vertexFunctionName = "forwardVertex",
            .fragmentFunctionName = "forwardFragment",
            .colorPixelFormat = metalDrawable->texture()->pixelFormat()
        };
        renderPipelines.createRenderPipeline(RenderPipelineType::ForwardDebug, debugConfig);
    }
    
    #pragma mark Min Max Depth Buffer
    {
        #pragma mark Init Min Max Depth Buffer Pipeline state
        {
            ComputePipelineConfig initMinMaxDepthConfig{
                .label = "Init Min Max Depth Buffer",
                .computeFunctionName = "initMinMaxDepthKernel"
            };
            renderPipelines.createComputePipeline(ComputePipelineType::InitMinMaxDepth, initMinMaxDepthConfig);
        }
        
        #pragma mark Min Max Depth Buffer Pipeline state
        {
            ComputePipelineConfig minMaxDepthConfig{
                .label = "Min Max Depth Buffer",
                .computeFunctionName = "minMaxDepthKernel"
            };
            renderPipelines.createComputePipeline(ComputePipelineType::MinMaxDepth, minMaxDepthConfig);
        }
    }
    
    #pragma mark Two Pass Blur
    {
        ComputePipelineConfig verticalBlurConfig{
            .label = "Vertical Blur",
            .computeFunctionName = "verticalBlurKernel"
        };
        renderPipelines.createComputePipeline(ComputePipelineType::VerticalBlur, verticalBlurConfig);
        
        ComputePipelineConfig horizontalBlurConfig{
            .label = "Vertical Blur",
            .computeFunctionName = "horizontalBlurKernel"
        };
        renderPipelines.createComputePipeline(ComputePipelineType::HorizontalBlur, horizontalBlurConfig);
    }
}

void Engine::createSphereGrid() {
    debug->clearLines();
    
    debugProbeCount = floor(metalLayer.drawableSize.width / (PROBE_SPACING * 1 << debugCascadeLevel)) * floor(metalLayer.drawableSize.height / (PROBE_SPACING * 1 << debugCascadeLevel));
    const float sphereRadius = 0.001f * float(1 << debugCascadeLevel) * PROBE_SPACING;
    simd::float3 sphereColor = {1.0f, 0.0f, 0.0f};
    std::vector<simd::float4> spherePositions;
    
    Probe* probes = reinterpret_cast<Probe*>(probePosBuffer[currentFrameIndex][debugCascadeLevel]->contents());
        
     for (int i = 0; i < debugProbeCount; ++i) {
         spherePositions.push_back(probes[i].position);
     }

    debug->drawSpheres(spherePositions, sphereRadius, sphereColor);
}

void Engine::createDebugLines() {
    
    rayCount = debugProbeCount * BASE_RAY * (1 << (2 * debugCascadeLevel));
    
    std::vector<simd::float4> startPoints;
    std::vector<simd::float4> endPoints;
    std::vector<simd::float4> colors;
    
    ProbeRay* rays = reinterpret_cast<ProbeRay*>(rayBuffer[currentFrameIndex][debugCascadeLevel]->contents());
    
    for (int i = 0; i < rayCount; i++) {
        startPoints.push_back(rays[i].intervalStart);
        endPoints.push_back(rays[i].intervalEnd);
        colors.push_back(rays[i].color);
    }

    debug->drawLines(startPoints, endPoints, colors);
}

void Engine::createViewRenderPassDescriptor() {
    // Define GBuffer formats
    albedoSpecularGBufferFormat = MTL::PixelFormatRGBA8Unorm_sRGB;
    normalMapGBufferFormat = MTL::PixelFormatRGBA8Snorm;
    depthGBufferFormat = MTL::PixelFormatR32Float;
    
    uint32_t width = metalLayer.drawableSize.width;
    uint32_t height = metalLayer.drawableSize.height;
        
    // GBuffer textures
    resourceManager->createGBufferTexture(width, height, albedoSpecularGBufferFormat, TextureName::AlbedoGBuffer);
    resourceManager->createGBufferTexture(width, height, normalMapGBufferFormat, TextureName::NormalGBuffer);
    resourceManager->createGBufferTexture(width, height, depthGBufferFormat, TextureName::DepthGBuffer);
    
    // Depth textures
    resourceManager->createDepthStencilTexture(width, height, TextureName::DepthStencilTexture);
    resourceManager->createDepthStencilTexture(width, height, TextureName::ForwardDepthStencilTexture);
    resourceManager->createRenderTargetTexture(width, height, MTL::PixelFormatR32Float, TextureName::LinearDepthTexture);
    
    // Raytracing textures
    resourceManager->createRaytracingOutputTexture(width, height, TextureName::FinalGatherTexture);
    resourceManager->createRaytracingOutputTexture(width, height, TextureName::BlurredColorTexture);
    
    // Intermediate blur texture
    MTL::TextureDescriptor* blurDesc = MTL::TextureDescriptor::alloc()->init();
    blurDesc->setTextureType(MTL::TextureType2D);
    blurDesc->setPixelFormat(MTL::PixelFormatRGBA16Float);
    blurDesc->setWidth(width);
    blurDesc->setHeight(height);
    blurDesc->setStorageMode(MTL::StorageModePrivate);
    blurDesc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    resourceManager->createTexture(blurDesc, TextureName::IntermediateBlurTexture);
    blurDesc->release();
    
    // Min-max depth texture
    MTL::TextureDescriptor* minMaxDescriptor = MTL::TextureDescriptor::alloc()->init();
    minMaxDescriptor->setTextureType(MTL::TextureType2D);
    minMaxDescriptor->setPixelFormat(MTL::PixelFormatRG32Float);
    minMaxDescriptor->setWidth(width);
    minMaxDescriptor->setHeight(height);
    minMaxDescriptor->setStorageMode(MTL::StorageModeShared);
    minMaxDescriptor->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    minMaxDescriptor->setMipmapLevelCount(log2(std::max(metalLayer.drawableSize.width, metalLayer.drawableSize.height)) + 1);
    resourceManager->createTexture(minMaxDescriptor, TextureName::MinMaxDepthTexture);
    minMaxDescriptor->release();
    
    // Create render pass descriptors
    viewRenderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
        
    // Set up render pass descriptor attachments
    viewRenderPassDescriptor->colorAttachments()->object(RenderTargetAlbedo)->setTexture(resourceManager->getTexture(TextureName::AlbedoGBuffer));
    viewRenderPassDescriptor->colorAttachments()->object(RenderTargetNormal)->setTexture(resourceManager->getTexture(TextureName::NormalGBuffer));
    viewRenderPassDescriptor->colorAttachments()->object(RenderTargetDepth)->setTexture(resourceManager->getTexture(TextureName::DepthGBuffer));
    viewRenderPassDescriptor->depthAttachment()->setTexture(resourceManager->getTexture(TextureName::DepthStencilTexture));
    viewRenderPassDescriptor->stencilAttachment()->setTexture(resourceManager->getTexture(TextureName::DepthStencilTexture));
    
    // Configure load/store actions
    viewRenderPassDescriptor->colorAttachments()->object(RenderTargetAlbedo)->setLoadAction(MTL::LoadActionClear);
    viewRenderPassDescriptor->colorAttachments()->object(RenderTargetAlbedo)->setStoreAction(MTL::StoreActionStore);
    viewRenderPassDescriptor->colorAttachments()->object(RenderTargetAlbedo)->setClearColor(MTL::ClearColor(0.0, 0.0, 0.0, 1.0));
    
    viewRenderPassDescriptor->colorAttachments()->object(RenderTargetNormal)->setLoadAction(MTL::LoadActionClear);
    viewRenderPassDescriptor->colorAttachments()->object(RenderTargetNormal)->setStoreAction(MTL::StoreActionStore);
    viewRenderPassDescriptor->colorAttachments()->object(RenderTargetNormal)->setClearColor(MTL::ClearColor(0.0, 0.0, 0.0, 1.0));
    
    viewRenderPassDescriptor->colorAttachments()->object(RenderTargetDepth)->setLoadAction(MTL::LoadActionClear);
    viewRenderPassDescriptor->colorAttachments()->object(RenderTargetDepth)->setStoreAction(MTL::StoreActionStore);
    viewRenderPassDescriptor->colorAttachments()->object(RenderTargetDepth)->setClearColor(MTL::ClearColor(1.0, 1.0, 1.0, 1.0));
    
    viewRenderPassDescriptor->depthAttachment()->setLoadAction(MTL::LoadActionDontCare);
    viewRenderPassDescriptor->depthAttachment()->setStoreAction(MTL::StoreActionDontCare);
    viewRenderPassDescriptor->depthAttachment()->setClearDepth(1.0);
    
    viewRenderPassDescriptor->stencilAttachment()->setLoadAction(MTL::LoadActionDontCare);
    viewRenderPassDescriptor->stencilAttachment()->setStoreAction(MTL::StoreActionDontCare);
    viewRenderPassDescriptor->stencilAttachment()->setClearStencil(0);
    
    forwardDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    finalGatherDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    depthPrepassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
}

void Engine::updateRenderPassDescriptor() {
    // Update G-buffer descriptor attachments
    viewRenderPassDescriptor->colorAttachments()->object(RenderTargetAlbedo)->setTexture(resourceManager->getTexture(TextureName::AlbedoGBuffer));
    viewRenderPassDescriptor->colorAttachments()->object(RenderTargetNormal)->setTexture(resourceManager->getTexture(TextureName::NormalGBuffer));
    viewRenderPassDescriptor->colorAttachments()->object(RenderTargetDepth)->setTexture(resourceManager->getTexture(TextureName::DepthGBuffer));
    viewRenderPassDescriptor->depthAttachment()->setTexture(resourceManager->getTexture(TextureName::DepthStencilTexture));
    viewRenderPassDescriptor->stencilAttachment()->setTexture(resourceManager->getTexture(TextureName::DepthStencilTexture));

    // Set up the final gather descriptor
    finalGatherDescriptor->colorAttachments()->object(0)->setTexture(metalDrawable->texture());
    finalGatherDescriptor->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
    finalGatherDescriptor->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);
    finalGatherDescriptor->colorAttachments()->object(0)->setClearColor(MTL::ClearColor(0.0, 0.4, 0.8, 1.0));
    
    // Set up the forward descriptor
    forwardDescriptor->colorAttachments()->object(0)->setTexture(metalDrawable->texture());
    forwardDescriptor->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionLoad);
    forwardDescriptor->depthAttachment()->setTexture(resourceManager->getTexture(TextureName::ForwardDepthStencilTexture));
    forwardDescriptor->depthAttachment()->setLoadAction(MTL::LoadActionLoad);
    forwardDescriptor->depthAttachment()->setClearDepth(1.0);
    forwardDescriptor->stencilAttachment()->setTexture(resourceManager->getTexture(TextureName::ForwardDepthStencilTexture));
    forwardDescriptor->stencilAttachment()->setLoadAction(MTL::LoadActionLoad);
    forwardDescriptor->stencilAttachment()->setClearStencil(0);

    // Set up the depth prepass descriptor
    depthPrepassDescriptor->colorAttachments()->object(0)->setTexture(resourceManager->getTexture(TextureName::LinearDepthTexture));
    depthPrepassDescriptor->depthAttachment()->setTexture(resourceManager->getTexture(TextureName::DepthStencilTexture));
    depthPrepassDescriptor->stencilAttachment()->setTexture(resourceManager->getTexture(TextureName::DepthStencilTexture));
}

void Engine::draw() {
    updateRenderPassDescriptor();
    MTL::CommandBuffer* commandBuffer = beginFrame(false);
    editor->beginFrame(forwardDescriptor);
    camera.position = editor->debug.cameraPosition;

    // Depth prepass
    renderPassManager->drawDepthPrepass(commandBuffer, meshes, frameDataBuffers[currentFrameIndex]);
    
    // G-Buffer pass
    MTL::RenderCommandEncoder* gBufferEncoder = commandBuffer->renderCommandEncoder(viewRenderPassDescriptor);
    gBufferEncoder->setLabel(NS::String::string("GBuffer", NS::ASCIIStringEncoding));
    if (gBufferEncoder) {
        renderPassManager->drawGBuffer(gBufferEncoder, meshes, frameDataBuffers[currentFrameIndex]);
        gBufferEncoder->endEncoding();
    }
    
    // Min max buffer is not used currently
    // renderPassManager->dispatchMinMaxDepthMipmaps(commandBuffer);
    
    renderPassManager->dispatchRaytracing(commandBuffer, 
                                         frameDataBuffers[currentFrameIndex], 
                                         cascadeDataBuffer[currentFrameIndex],
                                          probePosBuffer[currentFrameIndex][debugCascadeLevel],
                                          rayBuffer[currentFrameIndex][debugCascadeLevel]);

    // Final gathering pass
    MTL::RenderCommandEncoder* finalGatherEncoder = commandBuffer->renderCommandEncoder(finalGatherDescriptor);
    finalGatherEncoder->setLabel(NS::String::string("Final Gather", NS::ASCIIStringEncoding));
    if (finalGatherEncoder) {
        renderPassManager->drawFinalGathering(finalGatherEncoder, 
                                             frameDataBuffers[currentFrameIndex]);
        finalGatherEncoder->endEncoding();
    }

    // Debug visualization pass
    MTL::RenderCommandEncoder* debugEncoder = commandBuffer->renderCommandEncoder(forwardDescriptor);
    debugEncoder->setLabel(NS::String::string("Debug and ImGui", NS::ASCIIStringEncoding));
    renderPassManager->drawDebug(debugEncoder, commandBuffer, frameDataBuffers[currentFrameIndex]);
    debugEncoder->endEncoding();

    endFrame(commandBuffer);
}
