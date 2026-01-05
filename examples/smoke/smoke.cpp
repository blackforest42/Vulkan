/*
 * Vulkan Example - Smoke simulation with voxels and ray marching
 * example
 *
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#include "VulkanglTFModel.h"
#include "vulkanexamplebase.h"

// Vertex layout for this example
struct Vertex {
  float pos[3];
  float color[3];
};

class VulkanExample : public VulkanExampleBase {
 public:
  VkPhysicalDeviceVulkan13Features enabledFeatures13_{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};

  // resources for rendering the compute outputs
  struct Graphics {
    vks::Buffer cubeVerticesBuffer;
    vks::Buffer cubeIndicesBuffer;
    uint32_t indexCount{0};

    struct PreMarchUBO {
      alignas(16) glm::mat4 model;
      alignas(16) glm::mat4 cameraView;
      alignas(16) glm::mat4 perspective;
      alignas(16) glm::vec3 cameraPos;
      alignas(8) glm::vec2 screenRes;
      // toggle front and back face marching
      alignas(4) uint32_t enableFrontMarch{1};
    };

    struct RayMarchUBO {
      alignas(16) glm::mat4 cameraView;
      alignas(16) glm::vec3 cameraPos;
      alignas(8) glm::vec2 screenRes;
      float time{0};
    };

    struct UBO {
      PreMarchUBO preMarch;
      RayMarchUBO rayMarch;
    } ubos_;

    // vk Buffers
    struct UniformBuffers {
      vks::Buffer rayMarch;
      vks::Buffer preMarch;
    };
    std::array<UniformBuffers, MAX_CONCURRENT_FRAMES> uniformBuffers_;

    // Pipelines
    struct {
      // offscreen pass to generate entry/exit rays for ray marcher
      VkPipeline preMarch{VK_NULL_HANDLE};
      VkPipeline rayMarch{VK_NULL_HANDLE};
    } pipelines_{};
    struct {
      VkPipelineLayout preMarch{VK_NULL_HANDLE};
      VkPipelineLayout rayMarch{VK_NULL_HANDLE};
    } pipelineLayouts_;

    // Descriptors
    struct {
      VkDescriptorSetLayout preMarch{VK_NULL_HANDLE};
      VkDescriptorSetLayout rayMarch{VK_NULL_HANDLE};
    } descriptorSetLayouts_;
    struct DescriptorSets {
      VkDescriptorSet preMarch{VK_NULL_HANDLE};
      VkDescriptorSet rayMarch{VK_NULL_HANDLE};
    };
    std::array<DescriptorSets, MAX_CONCURRENT_FRAMES> descriptorSets_{};

    // Structure to hold offscreen velocity buffer
    struct VelocityBuffer {
      VkImage image;
      VkDeviceMemory memory;
      VkImageView imageView;
      VkDescriptorImageInfo descriptor;
      VkFormat format;
      VkExtent2D extent;
    };
    struct PreMarchPass {
      VelocityBuffer incoming;
      VelocityBuffer outgoing;
      VkSampler sampler;
    } preMarchPass_{};

  } graphics_;

  // Create a 4 channel 16-bit float velocity buffer
  void createVelocityBuffer(Graphics::VelocityBuffer& buffer) {
    // Note: VK_FORMAT_R16G16B16_SFLOAT has limited support, so we use RGBA16
    // The alpha channel will just be unused
    buffer.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    buffer.extent = {width_, height_};

    // Create image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = buffer.format;
    imageInfo.extent = {width_, height_, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT |  // For reading as texture
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;  // For copying/readback
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vkCreateImage(device_, &imageInfo, nullptr, &buffer.image);

    // Allocate device_-local memory
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device_, buffer.image, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;

    // Find device_-local memory type
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
      if ((memReqs.memoryTypeBits & (1 << i)) &&
          (memProps.memoryTypes[i].propertyFlags &
           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
        allocInfo.memoryTypeIndex = i;
        break;
      }
    }

    vkAllocateMemory(device_, &allocInfo, nullptr, &buffer.memory);
    vkBindImageMemory(device_, buffer.image, buffer.memory, 0);

    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = buffer.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = buffer.format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    vkCreateImageView(device_, &viewInfo, nullptr, &buffer.imageView);

    // descriptor
    buffer.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    buffer.descriptor.imageView = buffer.imageView;
    buffer.descriptor.sampler = graphics_.preMarchPass_.sampler;
  }

  void preparePreMarchPass() {
    // Create sampler to sample from the textures
    VkSamplerCreateInfo samplerCI = vks::initializers::samplerCreateInfo();
    samplerCI.magFilter = VK_FILTER_LINEAR;
    samplerCI.minFilter = VK_FILTER_LINEAR;
    samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = samplerCI.addressModeU;
    samplerCI.addressModeW = samplerCI.addressModeU;
    samplerCI.mipLodBias = 0.0f;
    samplerCI.maxAnisotropy = 1.0f;
    samplerCI.minLod = 0.0f;
    samplerCI.maxLod = 1.0f;
    samplerCI.unnormalizedCoordinates = VK_FALSE;
    samplerCI.compareEnable = VK_FALSE;
    samplerCI.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    VK_CHECK_RESULT(vkCreateSampler(device_, &samplerCI, nullptr,
                                    &graphics_.preMarchPass_.sampler));

    createVelocityBuffer(graphics_.preMarchPass_.incoming);
    createVelocityBuffer(graphics_.preMarchPass_.outgoing);
  }

  void setupDescriptors() {
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            /*total ubo count (across all pipelines) */ 2 *
                MAX_CONCURRENT_FRAMES),
    };

    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(
            poolSizes,
            /*total descriptor count*/ 2 * MAX_CONCURRENT_FRAMES);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo,
                                           nullptr, &descriptorPool_));

    // Layout: Ray march
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0 : Vertex shader uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding_id*/ 0),
    };

    VkDescriptorSetLayoutCreateInfo descriptorLayout =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorLayout, nullptr,
                                    &graphics_.descriptorSetLayouts_.rayMarch));

    // Layout: Pre march
    setLayoutBindings = {
        // Binding 0 : Fragment shader ubo
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0)};

    descriptorLayout =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorLayout, nullptr,
                                    &graphics_.descriptorSetLayouts_.preMarch));

    // Sets per frame, just like the buffers themselves
    // Images do not need to be duplicated per frame, we reuse the same one
    // for each frame
    for (auto i = 0; i < graphics_.uniformBuffers_.size(); i++) {
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool_, &graphics_.descriptorSetLayouts_.rayMarch, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &graphics_.descriptorSets_[i].rayMarch));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          // Binding 0 : Projection/View matrix as uniform buffer
          vks::initializers::writeDescriptorSet(
              graphics_.descriptorSets_[i].rayMarch,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0,
              &graphics_.uniformBuffers_[i].rayMarch.descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &graphics_.descriptorSetLayouts_.preMarch, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &graphics_.descriptorSets_[i].preMarch));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              graphics_.descriptorSets_[i].preMarch,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0,
              &graphics_.uniformBuffers_[i].preMarch.descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  void preparePipelines() {
    // Layout: Ray march
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &graphics_.descriptorSetLayouts_.rayMarch, 1);
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr,
                               &graphics_.pipelineLayouts_.rayMarch));

    // Layout: Pre march
    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &graphics_.descriptorSetLayouts_.preMarch, 2);
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr,
                               &graphics_.pipelineLayouts_.preMarch));

    // Pipeline
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
        vks::initializers::pipelineInputAssemblyStateCreateInfo(
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
    VkPipelineRasterizationStateCreateInfo rasterizationState =
        vks::initializers::pipelineRasterizationStateCreateInfo(
            VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE,
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
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_CULL_MODE,
    };
    VkPipelineDynamicStateCreateInfo dynamicState =
        vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo =
        vks::initializers::pipelineCreateInfo();

    // Pipeline: Pre march front
    pipelineCreateInfo.layout = graphics_.pipelineLayouts_.preMarch;
    shaderStages[0] = loadShader(getShadersPath() + "smoke/premarch.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "smoke/premarch.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);

    // Vertex bindings and attributes
    VkVertexInputBindingDescription vertexInputBinding = {
        vks::initializers::vertexInputBindingDescription(
            0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX)};
    std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
        vks::initializers::vertexInputAttributeDescription(
            0, 0, VK_FORMAT_R32G32B32_SFLOAT,
            offsetof(Vertex, pos)),  // Location 0 : Position
        vks::initializers::vertexInputAttributeDescription(
            0, 1, VK_FORMAT_R32G32B32_SFLOAT,
            offsetof(Vertex, color)),  // Location 1 : Color
    };
    VkPipelineVertexInputStateCreateInfo vertexInputStateCI =
        vks::initializers::pipelineVertexInputStateCreateInfo();
    vertexInputStateCI.vertexBindingDescriptionCount = 1;
    vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding;
    vertexInputStateCI.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(vertexInputAttributes.size());
    vertexInputStateCI.pVertexAttributeDescriptions =
        vertexInputAttributes.data();
    pipelineCreateInfo.pVertexInputState = &vertexInputStateCI;

    // Blend state
    blendAttachmentState.colorWriteMask = 0xF;
    blendAttachmentState.blendEnable = VK_FALSE;
    blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachmentState.dstColorBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

    depthStencilState.depthWriteEnable = VK_TRUE;

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
        &graphics_.preMarchPass_.incoming.format;
    pipelineRenderingCreateInfo.depthAttachmentFormat = depthFormat_;
    pipelineRenderingCreateInfo.stencilAttachmentFormat = depthFormat_;
    // Chain into the pipeline create info
    pipelineCreateInfo.pNext = &pipelineRenderingCreateInfo;

    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCreateInfo, nullptr,
                                              &graphics_.pipelines_.preMarch));

    // Pipeline: Ray march
    pipelineCreateInfo.layout = graphics_.pipelineLayouts_.rayMarch;
    VkPipelineVertexInputStateCreateInfo emptyInputState =
        vks::initializers::pipelineVertexInputStateCreateInfo();
    pipelineCreateInfo.pVertexInputState = &emptyInputState;
    shaderStages[0] = loadShader(getShadersPath() + "smoke/raymarch.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "smoke/raymarch.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);

    blendAttachmentState.blendEnable = VK_TRUE;
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &swapChain_.colorFormat_;
    rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;

    depthStencilState.depthTestEnable = VK_FALSE;
    depthStencilState.depthWriteEnable = VK_FALSE;

    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCreateInfo, nullptr,
                                              &graphics_.pipelines_.rayMarch));
  }

  // Prepare and initialize uniform buffer containing shader uniforms
  void prepareUniformBuffers() {
    for (auto& buffer : graphics_.uniformBuffers_) {
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.preMarch, sizeof(Graphics::PreMarchUBO),
          &graphics_.ubos_.preMarch));
      VK_CHECK_RESULT(buffer.preMarch.map());

      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.rayMarch, sizeof(Graphics::RayMarchUBO),
          &graphics_.ubos_.rayMarch));
      VK_CHECK_RESULT(buffer.rayMarch.map());
    }
  }

  void updateUniformBuffers() {
    // static buffers
    graphics_.ubos_.preMarch.model = glm::mat4(1.f);
    graphics_.ubos_.preMarch.cameraView = camera_.matrices_.view;
    graphics_.ubos_.preMarch.perspective = camera_.matrices_.perspective;
    graphics_.ubos_.preMarch.cameraPos = camera_.position_;
    graphics_.ubos_.preMarch.screenRes = glm::vec2(width_, height_);
    memcpy(graphics_.uniformBuffers_[currentBuffer_].preMarch.mapped,
           &graphics_.ubos_.preMarch, sizeof(Graphics::PreMarchUBO));

    graphics_.ubos_.rayMarch.cameraView = camera_.matrices_.view;
    graphics_.ubos_.rayMarch.screenRes = glm::vec2(width_, height_);
    graphics_.ubos_.rayMarch.cameraPos = camera_.position_;
    graphics_.ubos_.rayMarch.time =
        std::chrono::duration<float>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    memcpy(graphics_.uniformBuffers_[currentBuffer_].rayMarch.mapped,
           &graphics_.ubos_.rayMarch, sizeof(Graphics::RayMarchUBO));
  }

  void prepare() {
    VulkanExampleBase::prepare();
    prepareDynamicStates();
    generateCube();
    preparePreMarchPass();
    prepareUniformBuffers();
    setupDescriptors();
    preparePipelines();
    prepared_ = true;
  }

  virtual void render() {
    if (!prepared_) {
      return;
    }
    VulkanExampleBase::prepareFrame();
    updateUniformBuffers();
    buildCommandBuffer();
    VulkanExampleBase::submitFrame();
  }

  void buildCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    preMarchCmd(cmdBuffer);
    rayMarchCmd(cmdBuffer);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void preMarchCmd(VkCommandBuffer& cmdBuffer) {
    // With dynamic rendering there are no subpass dependencies, so we need to
    // take care of proper layout transitions by using barriers This set of
    // barriers prepares the color and depth images for output
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, swapChain_.images_[currentImageIndex_], 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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
    colorAttachment.imageView = graphics_.preMarchPass_.incoming.imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {0.0f, 0.0f, 0.0f, 0.0f};

    // A single depth stencil attachment info can be used, but they can also be
    // specified separately. When both are specified separately, the only
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

    vkCmdBeginRendering(cmdBuffer, &renderingInfo);

    // Set viewport and scissor
    VkViewport viewport{0.0f, 0.0f, (float)width_, (float)height_, 0.0f, 1.0f};
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics_.pipelines_.preMarch);
    vkCmdSetCullMode(cmdBuffer, VkCullModeFlagBits(VK_CULL_MODE_FRONT_BIT));

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            graphics_.pipelineLayouts_.preMarch, 0, 1,
                            &graphics_.descriptorSets_[currentBuffer_].preMarch,
                            0, nullptr);

    VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1,
                           &graphics_.cubeVerticesBuffer.buffer, offsets);
    vkCmdBindIndexBuffer(cmdBuffer, graphics_.cubeIndicesBuffer.buffer, 0,
                         VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmdBuffer, graphics_.indexCount, 1, 0, 0, 0);

    vkCmdEndRendering(cmdBuffer);
  }

  void rayMarchCmd(VkCommandBuffer& cmdBuffer) {
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
    vkCmdBeginRendering(cmdBuffer, &renderingInfo);
    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    VkDeviceSize offsets[1] = {0};

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            graphics_.pipelineLayouts_.rayMarch, 0, 1,
                            &graphics_.descriptorSets_[currentBuffer_].rayMarch,
                            0, nullptr);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics_.pipelines_.rayMarch);
    vkCmdSetCullMode(cmdBuffer, VkCullModeFlagBits(VK_CULL_MODE_NONE));

    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);
    drawUI(cmdBuffer);

    // End dynamic rendering
    vkCmdEndRendering(cmdBuffer);
  }

  void generateCube() {
    // Setup vertices indices for a colored cube
    std::vector<Vertex> vertices = {
        {{-1.0f, -1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
        {{1.0f, -1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
        {{1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
        {{-1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f}},
        {{-1.0f, -1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}},
        {{1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},
        {{1.0f, 1.0f, -1.0f}, {0.0f, 0.0f, 1.0f}},
        {{-1.0f, 1.0f, -1.0f}, {0.0f, 0.0f, 0.0f}},
    };

    std::vector<uint32_t> indices = {
        0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7,
        4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3,
    };

    graphics_.indexCount = static_cast<uint32_t>(indices.size());

    // Create buffers
    // For the sake of simplicity we won't stage the vertex data to the gpu
    // memory Vertex buffer
    VK_CHECK_RESULT(vulkanDevice_->createBuffer(
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &graphics_.cubeVerticesBuffer, vertices.size() * sizeof(Vertex),
        vertices.data()));
    // Index buffer
    VK_CHECK_RESULT(vulkanDevice_->createBuffer(
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &graphics_.cubeIndicesBuffer, indices.size() * sizeof(uint32_t),
        indices.data()));
  }

  virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay) {}

  void setupRenderPass() override {
    // With VK_KHR_dynamic_rendering we no longer need a render pass, so skip
    // the sample base render pass setup
    renderPass_ = VK_NULL_HANDLE;
  }

  void setupFrameBuffer() override {
    // With VK_KHR_dynamic_rendering we no longer need a frame buffer, so skip
    // the sample base framebuffer setup
  }

  void prepareDynamicStates() {}

  void getEnabledFeatures() override {}

  void getEnabledExtensions() override {}

  VulkanExample() : VulkanExampleBase() {
    title_ = "Smoke Simulation";
    camera_.type_ = Camera::CameraType::lookat;
    camera_.setMovementSpeed(25.f);
    camera_.setPosition(glm::vec3(0.0f, 0.0f, -3.f));
    camera_.setRotation(glm::vec3(0.0f, 15.0f, 0.0f));
    camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);

    apiVersion_ = VK_API_VERSION_1_3;
    enabledFeatures13_.dynamicRendering = VK_TRUE;
    deviceCreatepNextChain_ = &enabledFeatures13_;
  }

  ~VulkanExample() {
    if (device_) {
      vkDestroyPipeline(device_, graphics_.pipelines_.rayMarch, nullptr);
      vkDestroyPipeline(device_, graphics_.pipelines_.preMarch, nullptr);
      vkDestroyPipelineLayout(device_, graphics_.pipelineLayouts_.preMarch,
                              nullptr);
      vkDestroyPipelineLayout(device_, graphics_.pipelineLayouts_.rayMarch,
                              nullptr);
      vkDestroyDescriptorSetLayout(
          device_, graphics_.descriptorSetLayouts_.preMarch, nullptr);
      vkDestroyDescriptorSetLayout(
          device_, graphics_.descriptorSetLayouts_.rayMarch, nullptr);

      vkDestroyImageView(device_, graphics_.preMarchPass_.incoming.imageView,
                         nullptr);
      vkDestroyImage(device_, graphics_.preMarchPass_.incoming.image, nullptr);
      vkFreeMemory(device_, graphics_.preMarchPass_.incoming.memory, nullptr);

      vkDestroyImageView(device_, graphics_.preMarchPass_.outgoing.imageView,
                         nullptr);
      vkDestroyImage(device_, graphics_.preMarchPass_.outgoing.image, nullptr);
      vkFreeMemory(device_, graphics_.preMarchPass_.outgoing.memory, nullptr);
      vkDestroySampler(device_, graphics_.preMarchPass_.sampler, nullptr);
      for (auto& buffer : graphics_.uniformBuffers_) {
        buffer.rayMarch.destroy();
        buffer.preMarch.destroy();
      }
      graphics_.cubeVerticesBuffer.destroy();
      graphics_.cubeIndicesBuffer.destroy();
    }
  }
};

VULKAN_EXAMPLE_MAIN()
