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

// Offscreen frame buffer properties
// #define FB_COLOR_FORMAT VK_FORMAT_R8G8B8A8_UNORM
#define FB_COLOR_FORMAT VK_FORMAT_R16G16B16A16_SFLOAT

class VulkanExample : public VulkanExampleBase {
 public:
  vks::Texture cubeMap_{};
  vks::Texture colorMap_{};

  // Hacky workaround.
  // Boolean uniforms don't work in shaders for some reason.
  bool showBlackholeUI = true;
  bool gravatationalLensingEnabled = true;
  bool accDiskEnabled = true;
  bool accDiskParticleEnabled = true;
  bool toneMappingEnabled = true;

  struct BlackholeUBO {
    alignas(16) glm::mat4 cameraView;
    alignas(16) glm::vec3 cameraPos;
    alignas(8) glm::vec2 resolution;

    // Epoch time in seconds
    float time;

    int showBlackhole;
    int gravatationalLensingEnabled;
    int accDiskEnabled;
    int accDiskParticleEnabled;

    float accDiskHeight{0.55};
    float accDiskLit{0.25};
    float accDiskDensityV{2.0};
    float accDiskDensityH{4.0};
    float accDiskNoiseScale{.8};
    int accDiskNoiseLOD{5};
    float accDiskSpeed{0.5};
  };

  struct BloomUBO {
    // Tonemapping
    alignas(4) int tonemappingEnabled;
    alignas(4) float exposure{1.0f};
  };

  struct {
    BlackholeUBO blackhole;
    BloomUBO bloom;
  } ubos_;

  struct UniformBuffers {
    vks::Buffer bloom;
    vks::Buffer blackhole;
  };
  std::array<UniformBuffers, MAX_CONCURRENT_FRAMES> uniformBuffers_{};

  struct {
    VkPipelineLayout bloom;
    VkPipelineLayout blackhole;
  } pipelineLayouts_;

  struct {
    VkPipeline bloom;
    VkPipeline blackhole;
  } pipelines_;

  struct {
    VkDescriptorSetLayout bloom;
    VkDescriptorSetLayout blackhole;
  } descriptorSetLayouts_{};

  struct DescriptorSets {
    VkDescriptorSet bloom;
    VkDescriptorSet blackhole;
  };
  std::array<DescriptorSets, MAX_CONCURRENT_FRAMES> descriptorSets_{};

  // Framebuffer for offscreen rendering
  struct FrameBufferAttachment {
    VkImage image;
    VkDeviceMemory mem;
    VkImageView view;
  };
  struct FrameBuffer {
    VkFramebuffer framebuffer;
    FrameBufferAttachment color, depth;
    VkDescriptorImageInfo descriptor;
  };
  struct OffscreenPass {
    int32_t width, height;
    VkRenderPass renderPass;
    VkSampler sampler;
    FrameBuffer framebuffers;
  } offscreenPass_{};  // Handles the bloom pass

  VulkanExample() : VulkanExampleBase() {
    title = "Blackhole";
    camera_.type_ = Camera::CameraType::lookat;
    camera_.setPosition(glm::vec3(0.0f, 0.0f, -15.0f));
    camera_.setRotation(glm::vec3(0.0f));
    camera_.setRotationSpeed(0.25f);
    camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);
  }

  // (Part A.0) Called once in main() before renderLoop()
  void prepare() {
    VulkanExampleBase::prepare();
    loadAssets();
    prepareUniformBuffers();
    prepareOffscreen();
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
      // Blackhole
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.blackhole, sizeof(BlackholeUBO), &ubos_.blackhole));
      VK_CHECK_RESULT(buffer.blackhole.map());

      // Bloom
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.bloom, sizeof(BloomUBO), &ubos_.bloom));
      VK_CHECK_RESULT(buffer.bloom.map());
    }
  }

  // (A.3) Prepare the offscreen framebuffers used for the vertical- and
  // horizontal blur
  void prepareOffscreen() {
    offscreenPass_.width = width_;
    offscreenPass_.height = height_;

    // Find a suitable depth format
    VkFormat fbDepthFormat;
    VkBool32 validDepthFormat =
        vks::tools::getSupportedDepthFormat(physicalDevice_, &fbDepthFormat);
    assert(validDepthFormat);

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
    std::array<VkSubpassDependency, 3> dependencies{};

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask =
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dstAccessMask =
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    dependencies[0].dependencyFlags = 0;

    dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].dstSubpass = 0;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].dstStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[2].srcSubpass = 0;
    dependencies[2].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[2].srcStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[2].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[2].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[2].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

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

    // Create sampler to sample from the color attachments
    VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = sampler.addressModeU;
    sampler.addressModeW = sampler.addressModeU;
    sampler.mipLodBias = 0.0f;
    sampler.maxAnisotropy = 1.0f;
    sampler.minLod = 0.0f;
    sampler.maxLod = 1.0f;
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(
        vkCreateSampler(device_, &sampler, nullptr, &offscreenPass_.sampler));

    // Create one frame buffer
    prepareOffscreenFramebuffer(&offscreenPass_.framebuffers, FB_COLOR_FORMAT,
                                fbDepthFormat);
  }

  void prepareOffscreenFramebuffer(FrameBuffer* frameBuf,
                                   VkFormat colorFormat,
                                   VkFormat depthFormat) {
    // Color attachment
    VkImageCreateInfo image = vks::initializers::imageCreateInfo();
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = colorFormat;
    image.extent.width = width_;
    image.extent.height = height_;
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

    VkImageViewCreateInfo colorImageView =
        vks::initializers::imageViewCreateInfo();
    colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
    colorImageView.format = colorFormat;
    colorImageView.flags = 0;
    colorImageView.subresourceRange = {};
    colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorImageView.subresourceRange.baseMipLevel = 0;
    colorImageView.subresourceRange.levelCount = 1;
    colorImageView.subresourceRange.baseArrayLayer = 0;
    colorImageView.subresourceRange.layerCount = 1;

    VK_CHECK_RESULT(
        vkCreateImage(device_, &image, nullptr, &frameBuf->color.image));
    vkGetImageMemoryRequirements(device_, frameBuf->color.image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = vulkanDevice_->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(
        vkAllocateMemory(device_, &memAlloc, nullptr, &frameBuf->color.mem));
    VK_CHECK_RESULT(vkBindImageMemory(device_, frameBuf->color.image,
                                      frameBuf->color.mem, 0));

    colorImageView.image = frameBuf->color.image;
    VK_CHECK_RESULT(vkCreateImageView(device_, &colorImageView, nullptr,
                                      &frameBuf->color.view));

    // Depth stencil attachment
    image.format = depthFormat;
    image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageViewCreateInfo depthStencilView =
        vks::initializers::imageViewCreateInfo();
    depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format = depthFormat;
    depthStencilView.flags = 0;
    depthStencilView.subresourceRange = {};
    depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (vks::tools::formatHasStencil(depthFormat)) {
      depthStencilView.subresourceRange.aspectMask |=
          VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    depthStencilView.subresourceRange.baseMipLevel = 0;
    depthStencilView.subresourceRange.levelCount = 1;
    depthStencilView.subresourceRange.baseArrayLayer = 0;
    depthStencilView.subresourceRange.layerCount = 1;

    VK_CHECK_RESULT(
        vkCreateImage(device_, &image, nullptr, &frameBuf->depth.image));
    vkGetImageMemoryRequirements(device_, frameBuf->depth.image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = vulkanDevice_->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(
        vkAllocateMemory(device_, &memAlloc, nullptr, &frameBuf->depth.mem));
    VK_CHECK_RESULT(vkBindImageMemory(device_, frameBuf->depth.image,
                                      frameBuf->depth.mem, 0));

    depthStencilView.image = frameBuf->depth.image;
    VK_CHECK_RESULT(vkCreateImageView(device_, &depthStencilView, nullptr,
                                      &frameBuf->depth.view));

    VkImageView attachments[2]{frameBuf->color.view, frameBuf->depth.view};

    VkFramebufferCreateInfo fbufCreateInfo =
        vks::initializers::framebufferCreateInfo();
    fbufCreateInfo.renderPass = offscreenPass_.renderPass;
    fbufCreateInfo.attachmentCount = 2;
    fbufCreateInfo.pAttachments = attachments;
    fbufCreateInfo.width = width_;
    fbufCreateInfo.height = height_;
    fbufCreateInfo.layers = 1;

    VK_CHECK_RESULT(vkCreateFramebuffer(device_, &fbufCreateInfo, nullptr,
                                        &frameBuf->framebuffer));

    // Fill a descriptor for later use in a descriptor set
    frameBuf->descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    frameBuf->descriptor.imageView = frameBuf->color.view;
    frameBuf->descriptor.sampler = offscreenPass_.sampler;
  }

  // (A.4)
  void setupDescriptors() {
    // Pool
    // BUG: Magic numbers 8, 6, 4, up ahead. Removing them causes runtime error.
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              MAX_CONCURRENT_FRAMES * 8),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            MAX_CONCURRENT_FRAMES * 6)};
    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(poolSizes,
                                                    MAX_CONCURRENT_FRAMES * 4);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo,
                                           nullptr, &descriptorPool_));

    // Layout
    // Blackhole
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

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(
            setLayoutBindings.data(),
            static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.blackhole));

    // Bloom
    setLayoutBindings = {
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),  // Binding 0 : Fragment shader uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, 1)};
    descriptorSetLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI,
                                                nullptr,
                                                &descriptorSetLayouts_.bloom));

    // Image descriptor for the cube map texture
    VkDescriptorImageInfo textureDescriptor =
        vks::initializers::descriptorImageInfo(cubeMap_.sampler, cubeMap_.view,
                                               cubeMap_.imageLayout);
    // Image descriptor for the blackhole color texture
    VkDescriptorImageInfo blackholeColorTextureDescriptor =
        vks::initializers::descriptorImageInfo(
            colorMap_.sampler, colorMap_.view, colorMap_.imageLayout);

    for (auto i = 0; i < uniformBuffers_.size(); i++) {
      // Descriptor for blackhole
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool_, &descriptorSetLayouts_.blackhole, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].blackhole));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].blackhole, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              0, &uniformBuffers_[i].blackhole.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].blackhole,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &textureDescriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].blackhole,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2,
              &blackholeColorTextureDescriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Descriptor for bloom
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.bloom, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].bloom));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].bloom, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].bloom.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].bloom,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, /*binding id*/ 1,
              &offscreenPass_.framebuffers.descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  // (A.5)
  void preparePipelines() {
    // Layout
    VkPipelineLayoutCreateInfo pipelineLayoutCI =
        vks::initializers::pipelineLayoutCreateInfo(
            &descriptorSetLayouts_.blackhole, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.blackhole));
    pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.bloom, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.bloom));

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
        vks::initializers::pipelineCreateInfo(pipelineLayouts_.bloom,
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

    // Bloom pipeline
    shaderStages[0] = loadShader(getShadersPath() + "blackhole/bloom.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "blackhole/bloom.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);

    // No vertex input
    VkPipelineVertexInputStateCreateInfo emptyInputState =
        vks::initializers::pipelineVertexInputStateCreateInfo();
    pipelineCI.pVertexInputState = &emptyInputState;
    pipelineCI.layout = pipelineLayouts_.bloom;

    // Additive blending
    // blendAttachmentState.colorWriteMask = 0xF;
    // blendAttachmentState.blendEnable = VK_TRUE;
    // blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
    // blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    // blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    // blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
    // blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    // blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.bloom));

    // Blackhole pipeline
    pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState(
        {vkglTF::VertexComponent::Position, vkglTF::VertexComponent::UV,
         vkglTF::VertexComponent::Color, vkglTF::VertexComponent::Normal});
    pipelineCI.layout = pipelineLayouts_.blackhole;
    pipelineCI.renderPass = offscreenPass_.renderPass;
    shaderStages[0] =
        loadShader(getShadersPath() + "blackhole/blackhole.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] =
        loadShader(getShadersPath() + "blackhole/blackhole.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    blendAttachmentState.blendEnable = VK_FALSE;
    depthStencilState.depthWriteEnable = VK_TRUE;
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
    ubos_.blackhole.cameraView = camera_.matrices_.view;
    ubos_.blackhole.cameraPos = camera_.position_;
    ubos_.blackhole.time =
        std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    ubos_.blackhole.resolution =
        glm::vec2(offscreenPass_.width, offscreenPass_.height);
    ubos_.blackhole.showBlackhole = showBlackholeUI;
    ubos_.blackhole.gravatationalLensingEnabled = gravatationalLensingEnabled;
    ubos_.blackhole.accDiskEnabled = accDiskEnabled;
    ubos_.blackhole.accDiskParticleEnabled = accDiskParticleEnabled;
    memcpy(uniformBuffers_[currentBuffer_].blackhole.mapped, &ubos_.blackhole,
           sizeof(BlackholeUBO));

    ubos_.bloom.tonemappingEnabled = toneMappingEnabled;
    memcpy(uniformBuffers_[currentBuffer_].bloom.mapped, &ubos_.bloom,
           sizeof(BloomUBO));
  }

  // (B.2)
  void buildCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    // Blackhole
    {
      std::array<VkClearValue, 2> clearValues{};
      clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
      clearValues[1].depthStencil = {1.0f, 0};

      VkRenderPassBeginInfo renderPassBeginInfo =
          vks::initializers::renderPassBeginInfo();
      // renderPassBeginInfo.renderPass = renderPass_;
      // renderPassBeginInfo.framebuffer = frameBuffers_[currentImageIndex_];
      renderPassBeginInfo.renderPass = offscreenPass_.renderPass;
      renderPassBeginInfo.framebuffer = offscreenPass_.framebuffers.framebuffer;
      renderPassBeginInfo.renderArea.extent.width = offscreenPass_.width;
      renderPassBeginInfo.renderArea.extent.height = offscreenPass_.height;
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
                              pipelineLayouts_.blackhole, 0, 1,
                              &descriptorSets_[currentBuffer_].blackhole, 0,
                              nullptr);

      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipelines_.blackhole);
      vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

      vkCmdEndRenderPass(cmdBuffer);
    }

    // Bloom
    {
      VkClearValue clearValues[2]{};
      clearValues[0].color = {0.f, 0.0f, 0.0f};
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
      renderPassBeginInfo.pClearValues = clearValues;

      vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                           VK_SUBPASS_CONTENTS_INLINE);
      VkViewport viewport = vks::initializers::viewport(
          (float)width_, (float)height_, 0.0f, 1.0f);
      vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

      VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
      vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

      vkCmdBindDescriptorSets(
          cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.bloom, 0,
          1, &descriptorSets_[currentBuffer_].bloom, 0, nullptr);

      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipelines_.bloom);
      vkCmdDraw(cmdBuffer, 6, 1, 0, 0);
      drawUI(cmdBuffer);
      vkCmdEndRenderPass(cmdBuffer);
    }
    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay) {
    if (overlay->header("Settings")) {
      overlay->checkBox("Show Blackhole", &showBlackholeUI);
      overlay->checkBox("Show Gravitational Lensing",
                        &gravatationalLensingEnabled);
      overlay->checkBox("Accretion Disk Enabled", &accDiskEnabled);
      overlay->checkBox("Disk Particles Enabled", &accDiskParticleEnabled);
      overlay->sliderFloat("Accretion Disk Height",
                           &ubos_.blackhole.accDiskHeight, 0.0, 1.0);
      overlay->sliderFloat("Accretion Disk Intensity",
                           &ubos_.blackhole.accDiskLit, 0.0, 1.0);
      overlay->sliderFloat("Accretion Disk Density V",
                           &ubos_.blackhole.accDiskDensityV, 0.0, 3.0);
      overlay->sliderFloat("Accretion Disk Density H",
                           &ubos_.blackhole.accDiskDensityH, 0.0, 5.0);
      overlay->sliderFloat("Accretion Disk Noise Scale",
                           &ubos_.blackhole.accDiskNoiseScale, 0.1, 5.0);
      overlay->sliderInt("Accretion Disk LOD", &ubos_.blackhole.accDiskNoiseLOD,
                         0, 10);
      overlay->sliderFloat("Accretion Disk Speed",
                           &ubos_.blackhole.accDiskSpeed, 0.0, 2.0);

      overlay->checkBox("Tone Mapping Enabled", &toneMappingEnabled);
      overlay->sliderFloat("Exposure", &ubos_.bloom.exposure, 0.1, 10.0);
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
      vkDestroyPipeline(device_, pipelines_.bloom, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.blackhole, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.bloom, nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.blackhole,
                                   nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.bloom,
                                   nullptr);
      for (auto& buffer : uniformBuffers_) {
        buffer.blackhole.destroy();
      }
    }
  }
};

VULKAN_EXAMPLE_MAIN()