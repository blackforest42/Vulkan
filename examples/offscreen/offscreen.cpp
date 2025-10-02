/*
 * Vulkan Example - Offscreen rendering using a separate framebuffer
 *
 * Copyright (C) 2016-2025 Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#include "VulkanglTFModel.h"
#include "vulkanexamplebase.h"

// Offscreen frame buffer properties
#define FB_DIM 512
#define FB_COLOR_FORMAT VK_FORMAT_R8G8B8A8_UNORM

class VulkanExample : public VulkanExampleBase {
 public:
  bool debugDisplay = false;

  struct {
    vkglTF::Model example;
    vkglTF::Model plane;
  } models_;

  struct UniformData {
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 model;
    glm::vec4 lightPos = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
  } uniformData_;

  struct UniformBuffers {
    vks::Buffer model;
    vks::Buffer mirror;
    vks::Buffer offscreen;
  };
  std::array<UniformBuffers, MAX_CONCURRENT_FRAMES> uniformBuffers_;

  struct {
    VkPipelineLayout textured{VK_NULL_HANDLE};
    VkPipelineLayout shaded{VK_NULL_HANDLE};
  } pipelineLayouts_;

  struct {
    VkPipeline debug{VK_NULL_HANDLE};
    VkPipeline shaded{VK_NULL_HANDLE};
    VkPipeline shadedOffscreen{VK_NULL_HANDLE};
    VkPipeline mirror{VK_NULL_HANDLE};
  } pipelines_;

  struct {
    VkDescriptorSetLayout textured{VK_NULL_HANDLE};
    VkDescriptorSetLayout shaded{VK_NULL_HANDLE};
  } descriptorSetLayouts_;

  struct DescriptorSets {
    VkDescriptorSet offscreen{VK_NULL_HANDLE};
    VkDescriptorSet mirror{VK_NULL_HANDLE};
    VkDescriptorSet model{VK_NULL_HANDLE};
  };
  std::array<DescriptorSets, MAX_CONCURRENT_FRAMES> descriptorSets_;

  // Framebuffer for offscreen rendering
  struct FrameBufferAttachment {
    VkImage image;
    VkDeviceMemory mem;
    VkImageView view;
  };
  struct OffscreenPass {
    int32_t width, height;
    VkFramebuffer frameBuffer;
    FrameBufferAttachment color, depth;
    VkRenderPass renderPass;
    VkSampler sampler;
    VkDescriptorImageInfo descriptor;
  } offscreenPass_{};

  glm::vec3 modelPosition = glm::vec3(0.0f, -1.0f, 0.0f);
  glm::vec3 modelRotation = glm::vec3(0.0f);

  VulkanExample() : VulkanExampleBase() {
    title = "Offscreen rendering";
    timerSpeed *= 0.25f;
    camera_.type_ = Camera::CameraType::lookat;
    camera_.setPosition(glm::vec3(0.0f, 1.0f, -6.0f));
    camera_.setRotation(glm::vec3(-2.5f, 0.0f, 0.0f));
    camera_.setRotationSpeed(0.5f);
    camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);
    // The scene shader uses a clipping plane, so this feature has to be enabled
    enabledFeatures_.shaderClipDistance = VK_TRUE;
  }

  ~VulkanExample() {
    if (device_) {
      vkDestroyImageView(device_, offscreenPass_.color.view, nullptr);
      vkDestroyImage(device_, offscreenPass_.color.image, nullptr);
      vkFreeMemory(device_, offscreenPass_.color.mem, nullptr);
      vkDestroyImageView(device_, offscreenPass_.depth.view, nullptr);
      vkDestroyImage(device_, offscreenPass_.depth.image, nullptr);
      vkFreeMemory(device_, offscreenPass_.depth.mem, nullptr);
      vkDestroyRenderPass(device_, offscreenPass_.renderPass, nullptr);
      vkDestroySampler(device_, offscreenPass_.sampler, nullptr);
      vkDestroyFramebuffer(device_, offscreenPass_.frameBuffer, nullptr);
      vkDestroyPipeline(device_, pipelines_.debug, nullptr);
      vkDestroyPipeline(device_, pipelines_.shaded, nullptr);
      vkDestroyPipeline(device_, pipelines_.shadedOffscreen, nullptr);
      vkDestroyPipeline(device_, pipelines_.mirror, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.textured, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.shaded, nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.shaded,
                                   nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.textured,
                                   nullptr);
      for (auto& buffer : uniformBuffers_) {
        buffer.model.destroy();
        buffer.mirror.destroy();
        buffer.offscreen.destroy();
      }
    }
  }

  // Setup the offscreen framebuffer for rendering the mirrored scene
  // The color attachment of this framebuffer will then be used to sample from
  // in the fragment shader of the final pass
  void prepareOffscreen() {
    offscreenPass_.width = FB_DIM;
    offscreenPass_.height = FB_DIM;

    // Find a suitable depth format
    VkFormat fbDepthFormat;
    VkBool32 validDepthFormat =
        vks::tools::getSupportedDepthFormat(physicalDevice_, &fbDepthFormat);
    assert(validDepthFormat);

    // Color attachment
    VkImageCreateInfo image = vks::initializers::imageCreateInfo();
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = FB_COLOR_FORMAT;
    image.extent.width = offscreenPass_.width;
    image.extent.height = offscreenPass_.height;
    image.extent.depth = 1;
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    // We will sample directly from the color attachment
    image.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs;

    VK_CHECK_RESULT(
        vkCreateImage(device_, &image, nullptr, &offscreenPass_.color.image));
    vkGetImageMemoryRequirements(device_, offscreenPass_.color.image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = vulkanDevice_->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device_, &memAlloc, nullptr,
                                     &offscreenPass_.color.mem));
    VK_CHECK_RESULT(vkBindImageMemory(device_, offscreenPass_.color.image,
                                      offscreenPass_.color.mem, 0));

    VkImageViewCreateInfo colorImageView =
        vks::initializers::imageViewCreateInfo();
    colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
    colorImageView.format = FB_COLOR_FORMAT;
    colorImageView.subresourceRange = {};
    colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorImageView.subresourceRange.baseMipLevel = 0;
    colorImageView.subresourceRange.levelCount = 1;
    colorImageView.subresourceRange.baseArrayLayer = 0;
    colorImageView.subresourceRange.layerCount = 1;
    colorImageView.image = offscreenPass_.color.image;
    VK_CHECK_RESULT(vkCreateImageView(device_, &colorImageView, nullptr,
                                      &offscreenPass_.color.view));

    // Create sampler to sample from the attachment in the fragment shader
    VkSamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = samplerInfo.addressModeU;
    samplerInfo.addressModeW = samplerInfo.addressModeU;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(vkCreateSampler(device_, &samplerInfo, nullptr,
                                    &offscreenPass_.sampler));

    // Depth stencil attachment
    image.format = fbDepthFormat;
    image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VK_CHECK_RESULT(
        vkCreateImage(device_, &image, nullptr, &offscreenPass_.depth.image));
    vkGetImageMemoryRequirements(device_, offscreenPass_.depth.image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = vulkanDevice_->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device_, &memAlloc, nullptr,
                                     &offscreenPass_.depth.mem));
    VK_CHECK_RESULT(vkBindImageMemory(device_, offscreenPass_.depth.image,
                                      offscreenPass_.depth.mem, 0));

    VkImageViewCreateInfo depthStencilView =
        vks::initializers::imageViewCreateInfo();
    depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format = fbDepthFormat;
    depthStencilView.flags = 0;
    depthStencilView.subresourceRange = {};
    depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (fbDepthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
      depthStencilView.subresourceRange.aspectMask |=
          VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    depthStencilView.subresourceRange.baseMipLevel = 0;
    depthStencilView.subresourceRange.levelCount = 1;
    depthStencilView.subresourceRange.baseArrayLayer = 0;
    depthStencilView.subresourceRange.layerCount = 1;
    depthStencilView.image = offscreenPass_.depth.image;
    VK_CHECK_RESULT(vkCreateImageView(device_, &depthStencilView, nullptr,
                                      &offscreenPass_.depth.view));

    // Create a separate render pass for the offscreen rendering as it may
    // differ from the one used for scene rendering

    std::array<VkAttachmentDescription, 2> attchmentDescriptions = {};
    // Color attachment
    attchmentDescriptions[0].format = FB_COLOR_FORMAT;
    attchmentDescriptions[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attchmentDescriptions[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attchmentDescriptions[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attchmentDescriptions[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attchmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attchmentDescriptions[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attchmentDescriptions[0].finalLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // Depth attachment
    attchmentDescriptions[1].format = fbDepthFormat;
    attchmentDescriptions[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attchmentDescriptions[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attchmentDescriptions[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attchmentDescriptions[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attchmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attchmentDescriptions[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attchmentDescriptions[1].finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorReference = {
        0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthReference = {
        1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpassDescription = {};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorReference;
    subpassDescription.pDepthStencilAttachment = &depthReference;

    // Use subpass dependencies for layout transitions
    std::array<VkSubpassDependency, 2> dependencies;

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_NONE_KHR;
    dependencies[0].dstAccessMask =
        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask =
        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // Create the actual renderpass
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount =
        static_cast<uint32_t>(attchmentDescriptions.size());
    renderPassInfo.pAttachments = attchmentDescriptions.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpassDescription;
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();

    VK_CHECK_RESULT(vkCreateRenderPass(device_, &renderPassInfo, nullptr,
                                       &offscreenPass_.renderPass));

    VkImageView attachments[2];
    attachments[0] = offscreenPass_.color.view;
    attachments[1] = offscreenPass_.depth.view;

    VkFramebufferCreateInfo fbufCreateInfo =
        vks::initializers::framebufferCreateInfo();
    fbufCreateInfo.renderPass = offscreenPass_.renderPass;
    fbufCreateInfo.attachmentCount = 2;
    fbufCreateInfo.pAttachments = attachments;
    fbufCreateInfo.width = offscreenPass_.width;
    fbufCreateInfo.height = offscreenPass_.height;
    fbufCreateInfo.layers = 1;

    VK_CHECK_RESULT(vkCreateFramebuffer(device_, &fbufCreateInfo, nullptr,
                                        &offscreenPass_.frameBuffer));

    // Fill a descriptor for later use in a descriptor set
    offscreenPass_.descriptor.imageLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    offscreenPass_.descriptor.imageView = offscreenPass_.color.view;
    offscreenPass_.descriptor.sampler = offscreenPass_.sampler;
  }

  void loadAssets() {
    const uint32_t glTFLoadingFlags =
        vkglTF::FileLoadingFlags::PreTransformVertices |
        vkglTF::FileLoadingFlags::PreMultiplyVertexColors |
        vkglTF::FileLoadingFlags::FlipY;
    models_.plane.loadFromFile(getAssetPath() + "models/plane.gltf",
                               vulkanDevice_, queue_, glTFLoadingFlags);
    models_.example.loadFromFile(getAssetPath() + "models/chinesedragon.gltf",
                                 vulkanDevice_, queue_, glTFLoadingFlags);
  }

  void setupDescriptors() {
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              MAX_CONCURRENT_FRAMES * 3),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            MAX_CONCURRENT_FRAMES * 2)};
    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(poolSizes,
                                                    MAX_CONCURRENT_FRAMES * 3);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo,
                                           nullptr, &descriptorPool_));

    // Layout
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;
    VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo;

    // Shaded layout
    setLayoutBindings = {
        // Binding 0 : Vertex shader uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
    };
    descriptorLayoutInfo =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorLayoutInfo,
                                                nullptr,
                                                &descriptorSetLayouts_.shaded));

    // Textured layouts
    setLayoutBindings = {
        // Binding 0 : Vertex shader uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
        // Binding 1 : Fragment shader image sampler
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, 1)};
    descriptorLayoutInfo =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorLayoutInfo, nullptr,
                                    &descriptorSetLayouts_.textured));

    // Sets per frame, just like the buffers themselves
    // Images do not need to be duplicated per frame, we reuse the same one for
    // each frame
    for (auto i = 0; i < uniformBuffers_.size(); i++) {
      // Mirror plane descriptor set
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool_, &descriptorSetLayouts_.textured, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].mirror));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          // Binding 0 : Vertex shader uniform buffer
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].mirror, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
              &uniformBuffers_[i].mirror.descriptor),
          // Binding 1 : Fragment shader texture sampler
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].mirror,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
              &offscreenPass_.descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Shaded descriptor sets
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.shaded, 1);
      // Model
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].model));
      std::vector<VkWriteDescriptorSet> modelWriteDescriptorSets = {
          // Binding 0 : Vertex shader uniform buffer
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].model, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
              &uniformBuffers_[i].model.descriptor)};
      vkUpdateDescriptorSets(
          device_, static_cast<uint32_t>(modelWriteDescriptorSets.size()),
          modelWriteDescriptorSets.data(), 0, nullptr);

      // Offscreen
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].offscreen));
      std::vector<VkWriteDescriptorSet> offScreenWriteDescriptorSets = {
          // Binding 0 : Vertex shader uniform buffer
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].offscreen, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              0, &uniformBuffers_[i].offscreen.descriptor)};
      vkUpdateDescriptorSets(
          device_, static_cast<uint32_t>(offScreenWriteDescriptorSets.size()),
          offScreenWriteDescriptorSets.data(), 0, nullptr);
    }
  }

  void preparePipelines() {
    // Layouts
    VkPipelineLayoutCreateInfo pipelineLayoutInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &descriptorSetLayouts_.shaded, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutInfo,
                                           nullptr, &pipelineLayouts_.shaded));

    pipelineLayoutInfo = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.textured, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(
        device_, &pipelineLayoutInfo, nullptr, &pipelineLayouts_.textured));

    // Pipelines
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
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

    VkGraphicsPipelineCreateInfo pipelineCI =
        vks::initializers::pipelineCreateInfo(pipelineLayouts_.textured,
                                              renderPass_, 0);
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
        {vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Color,
         vkglTF::VertexComponent::Normal});

    rasterizationState.cullMode = VK_CULL_MODE_NONE;

    // Render-target debug display
    shaderStages[0] = loadShader(getShadersPath() + "offscreen/quad.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "offscreen/quad.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.debug));

    // Mirror
    shaderStages[0] = loadShader(getShadersPath() + "offscreen/mirror.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "offscreen/mirror.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.mirror));

    rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;

    // Phong shading pipelines
    pipelineCI.layout = pipelineLayouts_.shaded;
    // Scene
    shaderStages[0] = loadShader(getShadersPath() + "offscreen/phong.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "offscreen/phong.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.shaded));
    // Offscreen
    // Flip cull mode
    rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;
    pipelineCI.renderPass = offscreenPass_.renderPass;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.shadedOffscreen));
  }

  // Prepare and initialize uniform buffer containing shader uniforms
  void prepareUniformBuffers() {
    for (auto& buffer : uniformBuffers_) {
      // Mesh vertex shader uniform buffer block
      VK_CHECK_RESULT(
          vulkanDevice_->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      &buffer.model, sizeof(UniformData)));
      VK_CHECK_RESULT(buffer.model.map());
      // Mirror plane vertex shader uniform buffer block
      VK_CHECK_RESULT(
          vulkanDevice_->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      &buffer.mirror, sizeof(UniformData)));
      VK_CHECK_RESULT(buffer.mirror.map());
      // Offscreen vertex shader uniform buffer block
      VK_CHECK_RESULT(
          vulkanDevice_->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      &buffer.offscreen, sizeof(UniformData)));
      VK_CHECK_RESULT(buffer.offscreen.map());
    }
  }

  void updateUniformBuffers() {
    if (!paused) {
      modelRotation.y += frameTimer * 10.0f;
    }

    uniformData_.projection = camera_.matrices_.perspective;
    uniformData_.view = camera_.matrices_.view;

    // Model
    uniformData_.model = glm::mat4(1.0f);
    uniformData_.model =
        glm::rotate(uniformData_.model, glm::radians(modelRotation.y),
                    glm::vec3(0.0f, 1.0f, 0.0f));
    uniformData_.model = glm::translate(uniformData_.model, modelPosition);
    memcpy(uniformBuffers_[currentBuffer_].model.mapped, &uniformData_,
           sizeof(UniformData));

    // Mirror
    uniformData_.model = glm::mat4(1.0f);
    memcpy(uniformBuffers_[currentBuffer_].mirror.mapped, &uniformData_,
           sizeof(UniformData));

    uniformData_.model = glm::mat4(1.0f);
    uniformData_.model =
        glm::rotate(uniformData_.model, glm::radians(modelRotation.y),
                    glm::vec3(0.0f, 1.0f, 0.0f));
    uniformData_.model =
        glm::scale(uniformData_.model, glm::vec3(1.0f, -1.0f, 1.0f));
    uniformData_.model = glm::translate(uniformData_.model, modelPosition);
    memcpy(uniformBuffers_[currentBuffer_].offscreen.mapped, &uniformData_,
           sizeof(UniformData));
  }

  void updateUniformBufferOffscreen() {}

  void prepare() {
    VulkanExampleBase::prepare();
    loadAssets();
    prepareOffscreen();
    prepareUniformBuffers();
    setupDescriptors();
    preparePipelines();
    prepared_ = true;
  }

  void buildCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();

    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    /*
            First render pass: Offscreen rendering
    */
    {
      VkClearValue clearValues[2]{};
      clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
      clearValues[1].depthStencil = {1.0f, 0};

      VkRenderPassBeginInfo renderPassBeginInfo =
          vks::initializers::renderPassBeginInfo();
      renderPassBeginInfo.renderPass = offscreenPass_.renderPass;
      renderPassBeginInfo.framebuffer = offscreenPass_.frameBuffer;
      renderPassBeginInfo.renderArea.extent.width = offscreenPass_.width;
      renderPassBeginInfo.renderArea.extent.height = offscreenPass_.height;
      renderPassBeginInfo.clearValueCount = 2;
      renderPassBeginInfo.pClearValues = clearValues;

      vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                           VK_SUBPASS_CONTENTS_INLINE);

      VkViewport viewport =
          vks::initializers::viewport((float)offscreenPass_.width,
                                      (float)offscreenPass_.height, 0.0f, 1.0f);
      vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

      VkRect2D scissor = vks::initializers::rect2D(offscreenPass_.width,
                                                   offscreenPass_.height, 0, 0);
      vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

      // Mirrored scene
      vkCmdBindDescriptorSets(
          cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.shaded,
          0, 1, &descriptorSets_[currentBuffer_].offscreen, 0, nullptr);
      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipelines_.shadedOffscreen);
      models_.example.draw(cmdBuffer);

      vkCmdEndRenderPass(cmdBuffer);
    }

    /*
            Note: Explicit synchronization is not required between the render
       pass, as this is done implicit via sub pass dependencies
    */

    /*
            Second render pass: Scene rendering with applied radial blur
    */
    {
      VkClearValue clearValues[2]{};
      clearValues[0].color = defaultClearColor;
      clearValues[1].depthStencil = {1.0f, 0};

      VkRenderPassBeginInfo renderPassBeginInfo =
          vks::initializers::renderPassBeginInfo();
      renderPassBeginInfo.renderPass = renderPass_;
      renderPassBeginInfo.framebuffer = frameBuffers_[currentImageIndex_];
      renderPassBeginInfo.renderArea.extent.width = width_;
      renderPassBeginInfo.renderArea.extent.height = height_;
      renderPassBeginInfo.clearValueCount = 2;
      renderPassBeginInfo.pClearValues = clearValues;

      vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                           VK_SUBPASS_CONTENTS_INLINE);

      VkViewport viewport = vks::initializers::viewport(
          (float)width_, (float)height_, 0.0f, 1.0f);
      vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

      VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
      vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

      if (debugDisplay) {
        // Display the offscreen render target
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayouts_.textured, 0, 1,
                                &descriptorSets_[currentBuffer_].mirror, 0,
                                nullptr);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelines_.debug);
        vkCmdDraw(cmdBuffer, 3, 1, 0, 0);
      } else {
        // Render the scene
        // Reflection plane
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayouts_.textured, 0, 1,
                                &descriptorSets_[currentBuffer_].mirror, 0,
                                nullptr);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelines_.mirror);
        models_.plane.draw(cmdBuffer);
        // Model
        vkCmdBindDescriptorSets(
            cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.shaded,
            0, 1, &descriptorSets_[currentBuffer_].model, 0, nullptr);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelines_.shaded);
        models_.example.draw(cmdBuffer);
      }

      drawUI(cmdBuffer);

      vkCmdEndRenderPass(cmdBuffer);
    }

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  virtual void render() {
    if (!prepared_)
      return;
    VulkanExampleBase::prepareFrame();
    updateUniformBuffers();
    buildCommandBuffer();
    VulkanExampleBase::submitFrame();
  }

  virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay) {
    if (overlay->header("Settings")) {
      overlay->checkBox("Display render target", &debugDisplay);
    }
  }
};

VULKAN_EXAMPLE_MAIN()
