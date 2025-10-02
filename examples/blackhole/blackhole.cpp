/*
 * Vulkan Example - Cube map texture loading and displaying
 *
 * This sample shows how to load and render a cubemap. A cubemap is a textures
 * that contains 6 images, one per cube face. The sample displays the cubemap as
 * a skybox (background) and as a reflection on a selectable object
 *
 * Copyright (C) 2016-2025 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#include "vulkanexamplebase.h"

#include <ktx.h>
#include <ktxvulkan.h>
#include "VulkanglTFModel.h"
#include "stb_image.h"

class VulkanExample : public VulkanExampleBase {
 public:
  bool displaySkybox_ = true;
  bool enableBlackhole_ = true;

  vks::Texture cubeMap_{};
  vks::Texture colorMap_{};

  struct uniformData {
    glm::mat4 cameraView;
    alignas(16) glm::vec3 cameraPos;
    alignas(8) glm::vec2 resolution;
    float time;
    float exposure{1.0f};
    float gamma{2.2f};
    bool mouseControl = true;
  } uniformData_;
  std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers_;

  VkPipelineLayout pipelineLayout_{VK_NULL_HANDLE};
  struct {
    VkPipeline blackhole{VK_NULL_HANDLE};
  } pipelines_;

  VkDescriptorSetLayout descriptorSetLayout_{VK_NULL_HANDLE};
  std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSet_{};

  VulkanExample() : VulkanExampleBase() {
    title = "Blackhole";
    camera_.type_ = Camera::CameraType::lookat;
    camera_.setPosition(glm::vec3(0.0f, 0.0f, -10.0f));
    camera_.setRotation(glm::vec3(0.0f));
    camera_.setRotationSpeed(0.25f);
    camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);
  }

  // (Part A.0) Called once in main() before renderLoop()
  void prepare() {
    VulkanExampleBase::prepare();
    loadAssets();
    prepareUniformBuffers();
    setupDescriptors();
    preparePipelines();
    prepared_ = true;
  }

  // (A.1)
  void loadAssets() {
    uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices |
                                vkglTF::FileLoadingFlags::FlipY;
    // Cubemap texture
    loadCubemap(getAssetPath() + "textures/blackhole/skybox/cubemap.ktx",
                VK_FORMAT_R8G8B8A8_UNORM);
    loadTexture(getAssetPath() + "textures/blackhole/blackhole_color_map.ktx");
  }

  // (A.2)
  // Prepare and initialize uniform buffer containing shader uniforms
  void prepareUniformBuffers() {
    for (auto& buffer : uniformBuffers_) {
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer, sizeof(uniformData), &uniformData_));
      VK_CHECK_RESULT(buffer.map());
    }
  }

  // (A.3)
  void setupDescriptors() {
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              MAX_CONCURRENT_FRAMES),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_CONCURRENT_FRAMES)};
    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(poolSizes,
                                                    MAX_CONCURRENT_FRAMES);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo,
                                           nullptr, &descriptorPool_));

    // Layout
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0 : Fragment shader blackhole uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),

        // Binding 1 : Fragment shader cubemap
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, /*binding id*/ 1),
        // Binding 2 : Fragment shader blackhole 2D texture
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 2)};

    VkDescriptorSetLayoutCreateInfo descriptorLayout =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device_, &descriptorLayout, nullptr, &descriptorSetLayout_));

    // Image descriptor for the cube map texture
    VkDescriptorImageInfo textureDescriptor =
        vks::initializers::descriptorImageInfo(cubeMap_.sampler, cubeMap_.view,
                                               cubeMap_.imageLayout);
    // Image descriptor for the blackhole color texture
    VkDescriptorImageInfo blackholeColorTextureDescriptor =
        vks::initializers::descriptorImageInfo(
            colorMap_.sampler, colorMap_.view, colorMap_.imageLayout);

    for (auto i = 0; i < uniformBuffers_.size(); i++) {
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool_, &descriptorSetLayout_, 1);
      VK_CHECK_RESULT(
          vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSet_[i]));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSet_[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
              &uniformBuffers_[i].descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSet_[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
              &textureDescriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSet_[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2,
              &blackholeColorTextureDescriptor),
      };

      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  // (A.4)
  void preparePipelines() {
    // Layout
    VkPipelineLayoutCreateInfo pipelineLayoutCI =
        vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout_, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayout_));

    // Pipeline
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
        vks::initializers::pipelineInputAssemblyStateCreateInfo(
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
    VkPipelineRasterizationStateCreateInfo rasterizationState =
        vks::initializers::pipelineRasterizationStateCreateInfo(
            VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT,
            VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
    VkPipelineColorBlendAttachmentState blendAttachmentState =
        vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
    VkPipelineColorBlendStateCreateInfo colorBlendState =
        vks::initializers::pipelineColorBlendStateCreateInfo(
            1, &blendAttachmentState);
    VkPipelineDepthStencilStateCreateInfo depthStencilState =
        vks::initializers::pipelineDepthStencilStateCreateInfo(
            VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL);
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

    VkGraphicsPipelineCreateInfo pipelineCI =
        vks::initializers::pipelineCreateInfo(pipelineLayout_, renderPass_, 0);
    pipelineCI.pInputAssemblyState = &inputAssemblyState;
    pipelineCI.pRasterizationState = &rasterizationState;
    pipelineCI.pColorBlendState = &colorBlendState;
    pipelineCI.pMultisampleState = &multisampleState;
    pipelineCI.pViewportState = &viewportState;
    pipelineCI.pDepthStencilState = &depthStencilState;
    pipelineCI.pDynamicState = &dynamicState;
    pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCI.pStages = shaderStages.data();
    pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState(
        {vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal});

    // Enable depth test and write

    // Blackhole pipeline
    shaderStages[0] =
        loadShader(getShadersPath() + "blackhole/blackhole_main.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] =
        loadShader(getShadersPath() + "blackhole/blackhole_main.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.blackhole));
  }

  // (B) Called in VulkanExampleBase::renderLoop()
  virtual void render() {
    if (!prepared_)
      return;
    VulkanExampleBase::prepareFrame();
    updateUniformBuffers();
    buildCommandBuffer();
    VulkanExampleBase::submitFrame();
  }

  // (B.1)
  void updateUniformBuffers() {
    uniformData_.cameraView = camera_.matrices_.view;
    uniformData_.cameraPos = camera_.position_;
    uniformData_.time =
        std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    uniformData_.resolution = glm::vec2(width_, height_);
    memcpy(uniformBuffers_[currentBuffer_].mapped, &uniformData_,
           sizeof(uniformData_));
  }

  // (B.2)
  void buildCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    if (enableBlackhole_) {
      /*
              Render scene to offscreen framebuffer
      */

      std::array<VkClearValue, 2> clearValues{};
      clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
      clearValues[1].depthStencil = {1.0f, 0};

      VkRenderPassBeginInfo renderPassBeginInfo =
          vks::initializers::renderPassBeginInfo();
      renderPassBeginInfo.renderPass = renderPass_;
      renderPassBeginInfo.framebuffer = frameBuffers_[currentImageIndex_];
      renderPassBeginInfo.renderArea.offset.x = 0;
      renderPassBeginInfo.renderArea.offset.y = 0;
      renderPassBeginInfo.renderArea.extent.width = width_;
      renderPassBeginInfo.renderArea.extent.height = height_;
      renderPassBeginInfo.clearValueCount = 2;
      renderPassBeginInfo.pClearValues = clearValues.data();

      vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                           VK_SUBPASS_CONTENTS_INLINE);

      VkViewport viewport = vks::initializers::viewport(
          (float)width_, (float)height_, 0.0f, 1.0f);
      vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

      VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
      vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

      vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipelineLayout_, 0, 1,
                              &descriptorSet_[currentBuffer_], 0, nullptr);

      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipelines_.blackhole);
      vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

      vkCmdEndRenderPass(cmdBuffer);

    } else {
    }

    drawUI(cmdBuffer);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay) {
    if (overlay->header("Settings")) {
      overlay->checkBox("Skybox", &displaySkybox_);
      overlay->sliderFloat("Exposure", &uniformData_.exposure, 0.1, 10.0);
      overlay->sliderFloat("Gamma", &uniformData_.gamma, 1.0, 4.0);
    }
  }

  // Loads a cubemap from a file, uploads it to the device and create all
  // Vulkan resources required to display it
  void loadCubemap(std::string filename, VkFormat format) {
    ktxResult result;
    ktxTexture* ktxTexture;

    if (!vks::tools::fileExists(filename)) {
      vks::tools::exitFatal("Could not load texture from " + filename +
                                "\n\nMake sure the assets submodule has been "
                                "checked out and is up-to-date.",
                            -1);
    }
    result = ktxTexture_CreateFromNamedFile(
        filename.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
    assert(result == KTX_SUCCESS);

    // Get properties required for using and upload texture data from the ktx
    // texture object
    cubeMap_.width = ktxTexture->baseWidth;
    cubeMap_.height = ktxTexture->baseHeight;
    cubeMap_.mipLevels = ktxTexture->numLevels;
    ktx_uint8_t* ktxTextureData = ktxTexture_GetData(ktxTexture);
    ktx_size_t ktxTextureSize = ktxTexture_GetSize(ktxTexture);

    VkMemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs;

    // Create a host-visible staging buffer that contains the raw image data
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    VkBufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo();
    bufferCreateInfo.size = ktxTextureSize;
    // This buffer is used as a transfer source for the buffer copy
    bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK_RESULT(
        vkCreateBuffer(device_, &bufferCreateInfo, nullptr, &stagingBuffer));

    // Get memory requirements for the staging buffer (alignment, memory type
    // bits)
    vkGetBufferMemoryRequirements(device_, stagingBuffer, &memReqs);
    memAllocInfo.allocationSize = memReqs.size;
    // Get memory type index for a host visible buffer
    memAllocInfo.memoryTypeIndex = vulkanDevice_->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK_RESULT(
        vkAllocateMemory(device_, &memAllocInfo, nullptr, &stagingMemory));
    VK_CHECK_RESULT(
        vkBindBufferMemory(device_, stagingBuffer, stagingMemory, 0));

    // Copy texture data into staging buffer
    uint8_t* data;
    VK_CHECK_RESULT(
        vkMapMemory(device_, stagingMemory, 0, memReqs.size, 0, (void**)&data));
    memcpy(data, ktxTextureData, ktxTextureSize);
    vkUnmapMemory(device_, stagingMemory);

    // Create optimal tiled target image
    VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = format;
    imageCreateInfo.mipLevels = cubeMap_.mipLevels;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.extent = {cubeMap_.width, cubeMap_.height, 1};
    imageCreateInfo.usage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    // Cube faces count as array layers in Vulkan
    imageCreateInfo.arrayLayers = 6;
    // This flag is required for cube map images
    imageCreateInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

    VK_CHECK_RESULT(
        vkCreateImage(device_, &imageCreateInfo, nullptr, &cubeMap_.image));

    vkGetImageMemoryRequirements(device_, cubeMap_.image, &memReqs);

    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex = vulkanDevice_->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK_RESULT(vkAllocateMemory(device_, &memAllocInfo, nullptr,
                                     &cubeMap_.deviceMemory));
    VK_CHECK_RESULT(
        vkBindImageMemory(device_, cubeMap_.image, cubeMap_.deviceMemory, 0));

    VkCommandBuffer copyCmd = vulkanDevice_->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    // Setup buffer copy regions for each face including all of its miplevels
    std::vector<VkBufferImageCopy> bufferCopyRegions;
    uint32_t offset = 0;

    for (uint32_t face = 0; face < 6; face++) {
      for (uint32_t level = 0; level < cubeMap_.mipLevels; level++) {
        // Calculate offset into staging buffer for the current mip level and
        // face
        ktx_size_t offset;
        KTX_error_code ret =
            ktxTexture_GetImageOffset(ktxTexture, level, 0, face, &offset);
        assert(ret == KTX_SUCCESS);
        VkBufferImageCopy bufferCopyRegion = {};
        bufferCopyRegion.imageSubresource.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;
        bufferCopyRegion.imageSubresource.mipLevel = level;
        bufferCopyRegion.imageSubresource.baseArrayLayer = face;
        bufferCopyRegion.imageSubresource.layerCount = 1;
        bufferCopyRegion.imageExtent.width = ktxTexture->baseWidth >> level;
        bufferCopyRegion.imageExtent.height = ktxTexture->baseHeight >> level;
        bufferCopyRegion.imageExtent.depth = 1;
        bufferCopyRegion.bufferOffset = offset;
        bufferCopyRegions.push_back(bufferCopyRegion);
      }
    }

    // Image barrier for optimal image (target)
    // Set initial layout for all array layers (faces) of the optimal (target)
    // tiled texture
    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = cubeMap_.mipLevels;
    subresourceRange.layerCount = 6;

    vks::tools::setImageLayout(
        copyCmd, cubeMap_.image, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);

    // Copy the cube map faces from the staging buffer to the optimal tiled
    // image
    vkCmdCopyBufferToImage(copyCmd, stagingBuffer, cubeMap_.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           static_cast<uint32_t>(bufferCopyRegions.size()),
                           bufferCopyRegions.data());

    // Change texture image layout to shader read after all faces have been
    // copied
    cubeMap_.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vks::tools::setImageLayout(copyCmd, cubeMap_.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               cubeMap_.imageLayout, subresourceRange);

    vulkanDevice_->flushCommandBuffer(copyCmd, queue_, true);

    // Create sampler
    VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = sampler.addressModeU;
    sampler.addressModeW = sampler.addressModeU;
    sampler.mipLodBias = 0.0f;
    sampler.compareOp = VK_COMPARE_OP_NEVER;
    sampler.minLod = 0.0f;
    sampler.maxLod = static_cast<float>(cubeMap_.mipLevels);
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    sampler.maxAnisotropy = 1.0f;
    if (vulkanDevice_->features.samplerAnisotropy) {
      sampler.maxAnisotropy =
          vulkanDevice_->properties.limits.maxSamplerAnisotropy;
      sampler.anisotropyEnable = VK_TRUE;
    }
    VK_CHECK_RESULT(
        vkCreateSampler(device_, &sampler, nullptr, &cubeMap_.sampler));

    // Create image view
    VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
    // Cube map view type
    view.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    view.format = format;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    // 6 array layers (faces)
    view.subresourceRange.layerCount = 6;
    // Set number of mip levels
    view.subresourceRange.levelCount = cubeMap_.mipLevels;
    view.image = cubeMap_.image;
    VK_CHECK_RESULT(vkCreateImageView(device_, &view, nullptr, &cubeMap_.view));

    // Clean up staging resources
    vkFreeMemory(device_, stagingMemory, nullptr);
    vkDestroyBuffer(device_, stagingBuffer, nullptr);
    ktxTexture_Destroy(ktxTexture);
  }

  void loadTexture(const std::string& filename) {
    // We use the Khronos texture format
    // (https://www.khronos.org/opengles/sdk/tools/KTX/file_format_spec/)
    // Texture data contains 4 channels (RGBA) with unnormalized 8-bit values,
    // this is the most commonly supported format
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

    ktxResult result;
    ktxTexture* ktxTexture;

    if (!vks::tools::fileExists(filename)) {
      vks::tools::exitFatal("Could not load texture from " + filename +
                                "\n\nMake sure the assets submodule has been "
                                "checked out and is up-to-date.",
                            -1);
    }
    result = ktxTexture_CreateFromNamedFile(
        filename.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
    assert(result == KTX_SUCCESS);

    // Get properties required for using and upload texture data from the ktx
    // texture object
    colorMap_.width = ktxTexture->baseWidth;
    colorMap_.height = ktxTexture->baseHeight;
    colorMap_.mipLevels = ktxTexture->numLevels;
    ktx_uint8_t* ktxTextureData = ktxTexture_GetData(ktxTexture);
    ktx_size_t ktxTextureSize = ktxTexture_GetSize(ktxTexture);

    // We prefer using staging to copy the texture data to a device local
    // optimal image
    VkBool32 useStaging = true;

    // Only use linear tiling if forced
    bool forceLinearTiling = false;
    if (forceLinearTiling) {
      // Don't use linear if format is not supported for (linear) shader
      // sampling Get device properties for the requested texture format
      VkFormatProperties formatProperties;
      vkGetPhysicalDeviceFormatProperties(physicalDevice_, format,
                                          &formatProperties);
      useStaging = !(formatProperties.linearTilingFeatures &
                     VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
    }

    VkMemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs = {};

    if (useStaging) {
      // Copy data to an optimal tiled image
      // This loads the texture data into a host local buffer that is copied
      // to the optimal tiled image on the device

      // Create a host-visible staging buffer that contains the raw image data
      // This buffer will be the data source for copying texture data to the
      // optimal tiled image on the device
      VkBuffer stagingBuffer;
      VkDeviceMemory stagingMemory;

      VkBufferCreateInfo bufferCreateInfo =
          vks::initializers::bufferCreateInfo();
      bufferCreateInfo.size = ktxTextureSize;
      // This buffer is used as a transfer source for the buffer copy
      bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      VK_CHECK_RESULT(
          vkCreateBuffer(device_, &bufferCreateInfo, nullptr, &stagingBuffer));

      // Get memory requirements for the staging buffer (alignment, memory
      // type bits)
      vkGetBufferMemoryRequirements(device_, stagingBuffer, &memReqs);
      memAllocInfo.allocationSize = memReqs.size;
      // Get memory type index for a host visible buffer
      memAllocInfo.memoryTypeIndex = vulkanDevice_->getMemoryType(
          memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      VK_CHECK_RESULT(
          vkAllocateMemory(device_, &memAllocInfo, nullptr, &stagingMemory));
      VK_CHECK_RESULT(
          vkBindBufferMemory(device_, stagingBuffer, stagingMemory, 0));

      // Copy texture data into host local staging buffer
      uint8_t* data;
      VK_CHECK_RESULT(vkMapMemory(device_, stagingMemory, 0, memReqs.size, 0,
                                  (void**)&data));
      memcpy(data, ktxTextureData, ktxTextureSize);
      vkUnmapMemory(device_, stagingMemory);

      // Setup buffer copy regions for each mip level
      std::vector<VkBufferImageCopy> bufferCopyRegions;
      uint32_t offset = 0;

      for (uint32_t i = 0; i < colorMap_.mipLevels; i++) {
        // Calculate offset into staging buffer for the current mip level
        ktx_size_t offset;
        KTX_error_code ret =
            ktxTexture_GetImageOffset(ktxTexture, i, 0, 0, &offset);
        assert(ret == KTX_SUCCESS);
        // Setup a buffer image copy structure for the current mip level
        VkBufferImageCopy bufferCopyRegion = {};
        bufferCopyRegion.imageSubresource.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;
        bufferCopyRegion.imageSubresource.mipLevel = i;
        bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
        bufferCopyRegion.imageSubresource.layerCount = 1;
        bufferCopyRegion.imageExtent.width = ktxTexture->baseWidth >> i;
        bufferCopyRegion.imageExtent.height = ktxTexture->baseHeight >> i;
        bufferCopyRegion.imageExtent.depth = 1;
        bufferCopyRegion.bufferOffset = offset;
        bufferCopyRegions.push_back(bufferCopyRegion);
      }

      // Create optimal tiled target image on the device
      VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
      imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
      imageCreateInfo.format = format;
      imageCreateInfo.mipLevels = colorMap_.mipLevels;
      imageCreateInfo.arrayLayers = 1;
      imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
      imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
      imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      // Set initial layout of the image to undefined
      imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      imageCreateInfo.extent = {colorMap_.width, colorMap_.height, 1};
      imageCreateInfo.usage =
          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      VK_CHECK_RESULT(
          vkCreateImage(device_, &imageCreateInfo, nullptr, &colorMap_.image));

      vkGetImageMemoryRequirements(device_, colorMap_.image, &memReqs);
      memAllocInfo.allocationSize = memReqs.size;
      memAllocInfo.memoryTypeIndex = vulkanDevice_->getMemoryType(
          memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      VK_CHECK_RESULT(vkAllocateMemory(device_, &memAllocInfo, nullptr,
                                       &colorMap_.deviceMemory));
      VK_CHECK_RESULT(vkBindImageMemory(device_, colorMap_.image,
                                        colorMap_.deviceMemory, 0));

      VkCommandBuffer copyCmd = vulkanDevice_->createCommandBuffer(
          VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

      // Image memory barriers for the texture image

      // The sub resource range describes the regions of the image that will
      // be transitioned using the memory barriers below
      VkImageSubresourceRange subresourceRange = {};
      // Image only contains color data
      subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      // Start at first mip level
      subresourceRange.baseMipLevel = 0;
      // We will transition on all mip levels
      subresourceRange.levelCount = colorMap_.mipLevels;
      // The 2D texture only has one layer
      subresourceRange.layerCount = 1;

      // Transition the texture image layout to transfer target, so we can
      // safely copy our buffer data to it.
      VkImageMemoryBarrier imageMemoryBarrier =
          vks::initializers::imageMemoryBarrier();
      ;
      imageMemoryBarrier.image = colorMap_.image;
      imageMemoryBarrier.subresourceRange = subresourceRange;
      imageMemoryBarrier.srcAccessMask = 0;
      imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

      // Insert a memory dependency at the proper pipeline stages that will
      // execute the image layout transition Source pipeline stage is host
      // write/read execution (VK_PIPELINE_STAGE_HOST_BIT) Destination
      // pipeline stage is copy command execution
      // (VK_PIPELINE_STAGE_TRANSFER_BIT)
      vkCmdPipelineBarrier(copyCmd, VK_PIPELINE_STAGE_HOST_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                           nullptr, 1, &imageMemoryBarrier);

      // Copy mip levels from staging buffer
      vkCmdCopyBufferToImage(copyCmd, stagingBuffer, colorMap_.image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             static_cast<uint32_t>(bufferCopyRegions.size()),
                             bufferCopyRegions.data());

      // Once the data has been uploaded we transfer to the texture image to
      // the shader read layout, so it can be sampled from
      imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

      // Insert a memory dependency at the proper pipeline stages that will
      // execute the image layout transition Source pipeline stage is copy
      // command execution (VK_PIPELINE_STAGE_TRANSFER_BIT) Destination
      // pipeline stage fragment shader access
      // (VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
      vkCmdPipelineBarrier(copyCmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                           0, nullptr, 1, &imageMemoryBarrier);

      // Store current layout for later reuse
      colorMap_.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

      vulkanDevice_->flushCommandBuffer(copyCmd, queue_, true);

      // Clean up staging resources
      vkFreeMemory(device_, stagingMemory, nullptr);
      vkDestroyBuffer(device_, stagingBuffer, nullptr);
    } else {
      // Copy data to a linear tiled image

      VkImage mappableImage;
      VkDeviceMemory mappableMemory;

      // Load mip map level 0 to linear tiling image
      VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
      imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
      imageCreateInfo.format = format;
      imageCreateInfo.mipLevels = 1;
      imageCreateInfo.arrayLayers = 1;
      imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
      imageCreateInfo.tiling = VK_IMAGE_TILING_LINEAR;
      imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
      imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
      imageCreateInfo.extent = {colorMap_.width, colorMap_.height, 1};
      VK_CHECK_RESULT(
          vkCreateImage(device_, &imageCreateInfo, nullptr, &mappableImage));

      // Get memory requirements for this image like size and alignment
      vkGetImageMemoryRequirements(device_, mappableImage, &memReqs);
      // Set memory allocation size to required memory size
      memAllocInfo.allocationSize = memReqs.size;
      // Get memory type that can be mapped to host memory
      memAllocInfo.memoryTypeIndex = vulkanDevice_->getMemoryType(
          memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      VK_CHECK_RESULT(
          vkAllocateMemory(device_, &memAllocInfo, nullptr, &mappableMemory));
      VK_CHECK_RESULT(
          vkBindImageMemory(device_, mappableImage, mappableMemory, 0));

      // Map image memory
      void* data;
      VK_CHECK_RESULT(
          vkMapMemory(device_, mappableMemory, 0, memReqs.size, 0, &data));
      // Copy image data of the first mip level into memory
      memcpy(data, ktxTextureData, memReqs.size);
      vkUnmapMemory(device_, mappableMemory);

      // Linear tiled images don't need to be staged and can be directly used
      // as textures
      colorMap_.image = mappableImage;
      colorMap_.deviceMemory = mappableMemory;
      colorMap_.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

      // Setup image memory barrier transfer image to shader read layout
      VkCommandBuffer copyCmd = vulkanDevice_->createCommandBuffer(
          VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

      // The sub resource range describes the regions of the image we will be
      // transition
      VkImageSubresourceRange subresourceRange = {};
      subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      subresourceRange.baseMipLevel = 0;
      subresourceRange.levelCount = 1;
      subresourceRange.layerCount = 1;

      // Transition the texture image layout to shader read, so it can be
      // sampled from
      VkImageMemoryBarrier imageMemoryBarrier =
          vks::initializers::imageMemoryBarrier();
      ;
      imageMemoryBarrier.image = colorMap_.image;
      imageMemoryBarrier.subresourceRange = subresourceRange;
      imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
      imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

      // Insert a memory dependency at the proper pipeline stages that will
      // execute the image layout transition Source pipeline stage is host
      // write/read execution (VK_PIPELINE_STAGE_HOST_BIT) Destination
      // pipeline stage fragment shader access
      // (VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
      vkCmdPipelineBarrier(copyCmd, VK_PIPELINE_STAGE_HOST_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                           0, nullptr, 1, &imageMemoryBarrier);

      vulkanDevice_->flushCommandBuffer(copyCmd, queue_, true);
    }

    ktxTexture_Destroy(ktxTexture);

    // Create a texture sampler
    // In Vulkan textures are accessed by samplers
    // This separates all the sampling information from the texture data. This
    // means you could have multiple sampler objects for the same texture with
    // different settings Note: Similar to the samplers available with
    // OpenGL 3.3
    VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.mipLodBias = 0.0f;
    sampler.compareOp = VK_COMPARE_OP_NEVER;
    sampler.minLod = 0.0f;
    // Set max level-of-detail to mip level count of the texture
    sampler.maxLod = (useStaging) ? (float)colorMap_.mipLevels : 0.0f;
    // Enable anisotropic filtering
    // This feature is optional, so we must check if it's supported on the
    // device
    if (vulkanDevice_->features.samplerAnisotropy) {
      // Use max. level of anisotropy for this example
      sampler.maxAnisotropy =
          vulkanDevice_->properties.limits.maxSamplerAnisotropy;
      sampler.anisotropyEnable = VK_TRUE;
    } else {
      // The device does not support anisotropic filtering
      sampler.maxAnisotropy = 1.0;
      sampler.anisotropyEnable = VK_FALSE;
    }
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(
        vkCreateSampler(device_, &sampler, nullptr, &colorMap_.sampler));

    // Create image view
    // Textures are not directly accessed by the shaders and
    // are abstracted by image views containing additional
    // information and sub resource ranges
    VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    // The subresource range describes the set of mip levels (and array
    // layers) that can be accessed through this image view It's possible to
    // create multiple image views for a single image referring to different
    // (and/or overlapping) ranges of the image
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.baseMipLevel = 0;
    view.subresourceRange.baseArrayLayer = 0;
    view.subresourceRange.layerCount = 1;
    // Linear tiling usually won't support mip maps
    // Only set mip map count if optimal tiling is used
    view.subresourceRange.levelCount = (useStaging) ? colorMap_.mipLevels : 1;
    // The view will be based on the texture's image
    view.image = colorMap_.image;
    VK_CHECK_RESULT(
        vkCreateImageView(device_, &view, nullptr, &colorMap_.view));
  }

  // Enable physical device features required for this example
  virtual void getEnabledFeatures() {
    if (deviceFeatures_.samplerAnisotropy) {
      enabledFeatures_.samplerAnisotropy = VK_TRUE;
    }
  }

  ~VulkanExample() {
    if (device_) {
      vkDestroyImageView(device_, cubeMap_.view, nullptr);
      vkDestroyImage(device_, cubeMap_.image, nullptr);
      vkDestroySampler(device_, cubeMap_.sampler, nullptr);
      vkFreeMemory(device_, cubeMap_.deviceMemory, nullptr);
      vkDestroyPipeline(device_, pipelines_.blackhole, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
      for (auto& buffer : uniformBuffers_) {
        buffer.destroy();
      }
    }
  }
};

VULKAN_EXAMPLE_MAIN()