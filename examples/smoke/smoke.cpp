/*
 * Vulkan Example - Smoke simulation with voxels and ray marching
 * example
 *
 * Copyright (C) 2016-2025 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#include "VulkanglTFModel.h"
#include "vulkanexamplebase.h"

const int VOXEL_CUBOID_WIDTH = 10;
const int VOXEL_CUBOID_HEIGHT = 3;
const int VOXEL_INSTANCES =
    VOXEL_CUBOID_WIDTH * VOXEL_CUBOID_WIDTH * VOXEL_CUBOID_HEIGHT;
const float VOXEL_SCALE = .1f;

// Wrapper functions for aligned memory allocation
// There is currently no standard for this in C++ that works across all
// platforms and vendors, so we abstract this
void* alignedAlloc(size_t size, size_t alignment) {
  void* data = nullptr;
#if defined(_MSC_VER) || defined(__MINGW32__)
  data = _aligned_malloc(size, alignment);
#else
  int res = posix_memalign(&data, alignment, size);
  if (res != 0)
    data = nullptr;
#endif
  return data;
}

void alignedFree(void* data) {
#if defined(_MSC_VER) || defined(__MINGW32__)
  _aligned_free(data);
#else
  free(data);
#endif
}

class VulkanExample : public VulkanExampleBase {
 public:
  PFN_vkCmdBeginRenderingKHR vkCmdBeginRenderingKHR{VK_NULL_HANDLE};
  PFN_vkCmdEndRenderingKHR vkCmdEndRenderingKHR{VK_NULL_HANDLE};
  VkPhysicalDeviceDynamicRenderingFeaturesKHR
      enabledDynamicRenderingFeaturesKHR{};

  vkglTF::Model voxel_;

  // resources for rendering the compute outputs
  struct Graphics {
    struct UniformView {
      glm::mat4 projection;
      glm::mat4 view;
    } uniformView_;

    struct UniformModel {
      glm::mat4* model{nullptr};
    } uniformModel_;

    struct UniformBuffers {
      vks::Buffer view;
      vks::Buffer dynamic;
    };
    std::array<UniformBuffers, MAX_CONCURRENT_FRAMES> uniformBuffers_;

    size_t dynamicAlignment{0};
    VkPipeline pipeline_{VK_NULL_HANDLE};
    VkPipelineLayout pipelineLayout_{VK_NULL_HANDLE};
    VkDescriptorSetLayout descriptorSetLayout_{VK_NULL_HANDLE};
    std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSets_{};
  } graphics_;

  // Resources for the compute part of the example
  struct Compute {
    // Used to check if compute and graphics queue
    // families differ and require additional barriers
    uint32_t queueFamilyIndex;
    // Separate queue for compute commands (queue family may
    // differ from the one used for graphics)
    VkQueue queue;
    // Use a separate command pool (queue family may
    // differ from the one used for graphics)
    VkCommandPool commandPool;
    // Command buffer storing the dispatch commands and
    // barriers
    std::array<VkCommandBuffer, MAX_CONCURRENT_FRAMES> commandBuffers;
    // Compute shader binding layout
    VkDescriptorSetLayout descriptorSetLayout;
    // Compute shader bindings
    std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSets;
    // Fences to make sure command buffers are done
    std::array<VkFence, MAX_CONCURRENT_FRAMES> fences{};

    // Semaphores for submission ordering
    struct ComputeSemaphores {
      VkSemaphore ready{VK_NULL_HANDLE};
      VkSemaphore complete{VK_NULL_HANDLE};
    };

    std::array<ComputeSemaphores, MAX_CONCURRENT_FRAMES> semaphores{};

    // Layout of the compute pipeline
    VkPipelineLayout pipelineLayout;
    // Compute pipeline for N-Body velocity
    // calculation (1st pass)
    VkPipeline pipelineCalculate;
    // Compute pipeline for euler integration (2nd pass)
    VkPipeline pipelineIntegrate;

    // Compute shader uniform block object
    struct UniformData {
      // Frame delta time
      float deltaT{0.0f};
      int32_t particleCount{0};
      // Parameters used to control the behaviour of the particle system
      float gravity{0.002f};
      float power{0.75f};
      float soften{0.05f};
    } uniformView_;

    // Uniform buffer object containing particle system
    // parameters
    std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers;
  } compute_;

  void setupDescriptors() {
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              1 * MAX_CONCURRENT_FRAMES),
        // Dynamic uniform buffers require a different descriptor type
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            1 * MAX_CONCURRENT_FRAMES)};

    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(poolSizes,
                                                    1 * MAX_CONCURRENT_FRAMES);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo,
                                           nullptr, &descriptorPool_));

    // Layout
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0 : Vertex shader uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
        // Binding 1 : Dynamic uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            VK_SHADER_STAGE_VERTEX_BIT, 1)};
    VkDescriptorSetLayoutCreateInfo descriptorLayout =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device_, &descriptorLayout, nullptr, &graphics_.descriptorSetLayout_));

    // Sets per frame, just like the buffers themselves
    // Images do not need to be duplicated per frame, we reuse the same one
    // for each frame
    VkDescriptorSetAllocateInfo allocInfo =
        vks::initializers::descriptorSetAllocateInfo(
            descriptorPool_, &graphics_.descriptorSetLayout_, 1);
    for (auto i = 0; i < graphics_.uniformBuffers_.size(); i++) {
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &graphics_.descriptorSets_[i]));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          // Binding 0 : Projection/View matrix as uniform buffer
          vks::initializers::writeDescriptorSet(
              graphics_.descriptorSets_[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &graphics_.uniformBuffers_[i].view.descriptor),
          // Binding 1 : Instance matrix as dynamic uniform buffer
          vks::initializers::writeDescriptorSet(
              graphics_.descriptorSets_[i],
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, /*binding id*/ 1,
              &graphics_.uniformBuffers_[i].dynamic.descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  void preparePipelines() {
    // Layout
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &graphics_.descriptorSetLayout_, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo,
                                           nullptr,
                                           &graphics_.pipelineLayout_));

    // Pipeline
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
        vks::initializers::pipelineInputAssemblyStateCreateInfo(
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
    VkPipelineRasterizationStateCreateInfo rasterizationState =
        vks::initializers::pipelineRasterizationStateCreateInfo(
            VK_POLYGON_MODE_LINE, VK_CULL_MODE_NONE,
            VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
    VkPipelineColorBlendAttachmentState blendAttachmentState =
        vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
    VkPipelineColorBlendStateCreateInfo colorBlendState =
        vks::initializers::pipelineColorBlendStateCreateInfo(
            1, &blendAttachmentState);
    VkPipelineDepthStencilStateCreateInfo depthStencilState =
        vks::initializers::pipelineDepthStencilStateCreateInfo(
            VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
    VkPipelineViewportStateCreateInfo viewportState =
        vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
    VkPipelineMultisampleStateCreateInfo multisampleState =
        vks::initializers::pipelineMultisampleStateCreateInfo(
            VK_SAMPLE_COUNT_1_BIT, 0);
    std::vector<VkDynamicState> dynamicStateEnables = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState =
        vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

    // Shaders
    shaderStages[0] = loadShader(getShadersPath() + "smoke/smoke.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "smoke/smoke.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);

    VkGraphicsPipelineCreateInfo pipelineCreateInfo =
        vks::initializers::pipelineCreateInfo(graphics_.pipelineLayout_,
                                              renderPass_, 0);
    pipelineCreateInfo.pVertexInputState =
        vkglTF::Vertex::getPipelineVertexInputState(
            {vkglTF::VertexComponent::Position, vkglTF::VertexComponent::UV,
             vkglTF::VertexComponent::Color, vkglTF::VertexComponent::Normal});
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pColorBlendState = &colorBlendState;
    pipelineCreateInfo.pMultisampleState = &multisampleState;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pDepthStencilState = &depthStencilState;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCreateInfo.pStages = shaderStages.data();

    // New create info to define color, depth and stencil attachments at
    // pipeline create time
    VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo{};
    pipelineRenderingCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    pipelineRenderingCreateInfo.colorAttachmentCount = 1;
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &swapChain_.colorFormat_;
    pipelineRenderingCreateInfo.depthAttachmentFormat = depthFormat_;
    pipelineRenderingCreateInfo.stencilAttachmentFormat = depthFormat_;
    // Chain into the pipeline creat einfo
    pipelineCreateInfo.pNext = &pipelineRenderingCreateInfo;

    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCreateInfo, nullptr,
                                              &graphics_.pipeline_));
  }

  // Prepare and initialize uniform buffer containing shader uniforms
  void prepareUniformBuffers() {
    // Allocate data for the dynamic uniform buffer object
    // We allocate this manually as the alignment of the offset differs
    // between GPUs

    // Calculate required alignment based on minimum device offset alignment
    graphics_.dynamicAlignment = sizeof(glm::mat4);

    size_t bufferSize = VOXEL_INSTANCES * graphics_.dynamicAlignment;

    graphics_.uniformModel_.model =
        (glm::mat4*)alignedAlloc(bufferSize, graphics_.dynamicAlignment);
    assert(graphics_.uniformModel_.model);

    for (auto& buffer : graphics_.uniformBuffers_) {
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.view, sizeof(Graphics::UniformView),
          &graphics_.uniformView_));

      // Uniform buffer object with per-object matrices
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &buffer.dynamic, bufferSize));

      // Override descriptor range to [base, base + dynamicAlignment]
      buffer.dynamic.descriptor.range = graphics_.dynamicAlignment;

      // Map persistent
      VK_CHECK_RESULT(buffer.view.map());
      VK_CHECK_RESULT(buffer.dynamic.map());
    }
  }

  void updateUniformBuffers() {
    graphics_.uniformView_.projection = camera_.matrices_.perspective;
    graphics_.uniformView_.view = camera_.matrices_.view;
    memcpy(graphics_.uniformBuffers_[currentBuffer_].view.mapped,
           &graphics_.uniformView_, sizeof(Graphics::UniformView));
  }

  void prepareVoxelPositions() {
    // Dynamic ubo with per-object model matrices indexed by offsets in the
    // command buffer
    for (uint32_t y = 0; y < VOXEL_CUBOID_HEIGHT; y++) {
      for (uint32_t x = 0; x < VOXEL_CUBOID_WIDTH; x++) {
        for (uint32_t z = 0; z < VOXEL_CUBOID_WIDTH; z++) {
          const uint32_t index = y * VOXEL_CUBOID_WIDTH * VOXEL_CUBOID_WIDTH +
                                 x * VOXEL_CUBOID_WIDTH + z;

          // Aligned offset
          glm::mat4* modelMat =
              (glm::mat4*)(((uint64_t)graphics_.uniformModel_.model +
                            (index * graphics_.dynamicAlignment)));

          glm::vec3 voxel_size = glm::abs(voxel_.dimensions.size);

          // Update matrices
          glm::vec3 pos =
              glm::vec3(x * voxel_size.x, y * voxel_size.y, z * voxel_size.z);
          *modelMat = glm::translate(
              glm::scale(glm::mat4(1.0), glm::vec3(VOXEL_SCALE)), pos);
        }
      }
    }
    memcpy(graphics_.uniformBuffers_[currentBuffer_].dynamic.mapped,
           graphics_.uniformModel_.model,
           graphics_.uniformBuffers_[currentBuffer_].dynamic.size);
    // Flush to make changes visible to the host
    VkMappedMemoryRange memoryRange = vks::initializers::mappedMemoryRange();
    memoryRange.memory =
        graphics_.uniformBuffers_[currentBuffer_].dynamic.memory;
    memoryRange.size = graphics_.uniformBuffers_[currentBuffer_].dynamic.size;
    vkFlushMappedMemoryRanges(device_, 1, &memoryRange);
  }

  void prepareDyanmicRendering() {
    // Since we use an extension, we need to expliclity load the function
    // pointers for extension related Vulkan commands
    vkCmdBeginRenderingKHR = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(
        vkGetDeviceProcAddr(device_, "vkCmdBeginRenderingKHR"));
    vkCmdEndRenderingKHR = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(
        vkGetDeviceProcAddr(device_, "vkCmdEndRenderingKHR"));
  }

  void setupRenderPass() override {
    // With VK_KHR_dynamic_rendering we no longer need a render pass, so skip
    // the sample base render pass setup
    renderPass_ = VK_NULL_HANDLE;
  }

  void setupFrameBuffer() override {
    // With VK_KHR_dynamic_rendering we no longer need a frame buffer, so skip
    // the sample base framebuffer setup
  }

  void getEnabledFeatures() override {
    enabledFeatures_.fillModeNonSolid = deviceFeatures_.fillModeNonSolid;
  }

  void prepare() {
    VulkanExampleBase::prepare();
    prepareDyanmicRendering();
    loadAssets();
    prepareUniformBuffers();
    prepareVoxelPositions();
    setupDescriptors();
    preparePipelines();
    prepared_ = true;
  }

  void buildCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, swapChain_.images_[currentImageIndex_], 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, depthStencil_.image, 0,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VkImageSubresourceRange{
            VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0,
            1});

    // New structures are used to define the attachments used in dynamic
    // rendering
    VkRenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachment.imageView = swapChain_.imageViews_[currentImageIndex_];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {0.0f, 0.0f, 0.0f, 0.0f};

    // A single depth stencil attachment info can be used, but they can also
    // be specified separately. When both are specified separately, the only
    // requirement is that the image view is identical.
    VkRenderingAttachmentInfoKHR depthStencilAttachment{};
    depthStencilAttachment.sType =
        VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    depthStencilAttachment.imageView = depthStencil_.view;
    depthStencilAttachment.imageLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthStencilAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthStencilAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthStencilAttachment.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfoKHR renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    renderingInfo.renderArea = {0, 0, width_, height_};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthStencilAttachment;
    renderingInfo.pStencilAttachment = &depthStencilAttachment;

    // Begin dynamic rendering
    vkCmdBeginRenderingKHR(cmdBuffer, &renderingInfo);
    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics_.pipeline_);

    for (uint32_t j = 0; j < VOXEL_INSTANCES; j++) {
      uint32_t dynamicOffset =
          j * static_cast<uint32_t>(graphics_.dynamicAlignment);
      vkCmdBindDescriptorSets(
          cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_.pipelineLayout_,
          0, 1, &graphics_.descriptorSets_[currentBuffer_], 1, &dynamicOffset);
      voxel_.draw(cmdBuffer);
    }
    drawUI(cmdBuffer);
    // End dynamic rendering
    vkCmdEndRenderingKHR(cmdBuffer);

    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, swapChain_.images_[currentImageIndex_],
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void loadAssets() {
    const uint32_t glTFLoadingFlags =
        vkglTF::FileLoadingFlags::PreTransformVertices |
        vkglTF::FileLoadingFlags::PreMultiplyVertexColors |
        vkglTF::FileLoadingFlags::FlipY;
    voxel_.loadFromFile(getAssetPath() + "models/cube.gltf", vulkanDevice_,
                        queue_, glTFLoadingFlags);
  }

  virtual void render() {
    if (!prepared_)
      return;
    VulkanExampleBase::prepareFrame();
    updateUniformBuffers();
    buildCommandBuffer();
    VulkanExampleBase::submitFrame();
  }

  virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay) {}

  VulkanExample() : VulkanExampleBase() {
    title = "Smoke Simulation";
    camera_.type_ = Camera::CameraType::firstperson;
    camera_.setMovementSpeed(25.f);
    camera_.setPosition(glm::vec3(0.0f, 0.0f, -16.f));
    camera_.setRotation(glm::vec3(0.0f, 15.0f, 0.0f));
    camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);

    enabledInstanceExtensions_.push_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    // The sample uses the extension (instead of Vulkan 1.2, where dynamic
    // rendering is core)
    enabledDeviceExtensions_.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    enabledDeviceExtensions_.push_back(VK_KHR_MAINTENANCE2_EXTENSION_NAME);
    enabledDeviceExtensions_.push_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);
    enabledDeviceExtensions_.push_back(
        VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
    enabledDeviceExtensions_.push_back(
        VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);

    // in addition to the extension, the feature needs to be explicitly
    // enabled too by chaining the extension structure into device creation
    enabledDynamicRenderingFeaturesKHR.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;
    enabledDynamicRenderingFeaturesKHR.dynamicRendering = VK_TRUE;

    deviceCreatepNextChain_ = &enabledDynamicRenderingFeaturesKHR;
  }

  ~VulkanExample() {
    if (device_) {
      vkDestroyPipeline(device_, graphics_.pipeline_, nullptr);
      vkDestroyPipelineLayout(device_, graphics_.pipelineLayout_, nullptr);
      vkDestroyDescriptorSetLayout(device_, graphics_.descriptorSetLayout_,
                                   nullptr);
      for (auto& buffer : graphics_.uniformBuffers_) {
        buffer.view.destroy();
        buffer.dynamic.destroy();
      }
    }
  }
};

VULKAN_EXAMPLE_MAIN()
