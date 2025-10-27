/*
 * Vulkan Example - Fluid simulation
 *
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
#define FB_COLOR_FORMAT VK_FORMAT_R32G32B32A32_SFLOAT

class VulkanExample : public VulkanExampleBase {
 public:
  struct AdvectionUBO {
    glm::vec2 viewportResolution;
    float timestep;
  };
  struct {
    AdvectionUBO advection;
  } ubos_;

  struct UniformBuffers {
    vks::Buffer advection;
  };
  std::array<UniformBuffers, MAX_CONCURRENT_FRAMES> uniformBuffers_{};

  struct {
    VkDescriptorSetLayout advection;
  } descriptorSetLayouts_{};

  struct DescriptorSets {
    VkDescriptorSet advection;
  };
  std::array<DescriptorSets, MAX_CONCURRENT_FRAMES> descriptorSets_{};

  struct {
    VkPipelineLayout advection;
  } pipelineLayouts_{};

  struct {
    VkPipeline advection;
  } pipelines_{};

  struct FrameBufferAttachment {
    VkImage image;
    VkDeviceMemory mem;
    VkImageView view;
  };
  struct FrameBuffer {
    int32_t width, height;
    VkFramebuffer framebuffer;
    FrameBufferAttachment color;
    VkDescriptorImageInfo descriptor;
  };
  struct OffscreenPass {
    VkRenderPass renderPass{};
    VkSampler sampler{};
  } offscreenPass_;

  // 2 framebuffers for each field, index 0 is for reading, 1 is for writing
  std::array<FrameBuffer, 2> velocity_field_{};
  std::array<FrameBuffer, 2> pressure_field_{};
  std::array<FrameBuffer, 1> divergence_{};

  VulkanExample() {
    title = "Blackhole";
    camera_.type_ = Camera::CameraType::lookat;
    camera_.setPosition(glm::vec3(0.0f, 0.0f, -15.0f));
    camera_.setRotation(glm::vec3(0.0f));
    camera_.setRotationSpeed(0.25f);
    camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);
  }

  // (Part A)
  void prepare() override {
    VulkanExampleBase::prepare();
    prepareUniformBuffers();
    prepareOffscreen();
    setupDescriptors();
    preparePipelines();
    prepared_ = true;
  }

  void prepareUniformBuffers() {
    for (auto& buffer : uniformBuffers_) {
      // Blackhole
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.advection, sizeof(AdvectionUBO), &ubos_.advection));
      VK_CHECK_RESULT(buffer.advection.map());
    }
  }

  void prepareOffscreen() {
    // Create a separate render pass for the offscreen rendering as it may
    // differ from the one used for scene rendering
    VkAttachmentDescription attchmentDescriptions;
    // Color attachment
    attchmentDescriptions.format = FB_COLOR_FORMAT;
    attchmentDescriptions.samples = VK_SAMPLE_COUNT_1_BIT;
    attchmentDescriptions.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attchmentDescriptions.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attchmentDescriptions.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attchmentDescriptions.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attchmentDescriptions.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attchmentDescriptions.finalLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorReference = {
        0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpassDescription = {};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorReference;

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
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &attchmentDescriptions;
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

    for (auto& fb : velocity_field_) {
      prepareOffscreenFramebuffer(&fb, FB_COLOR_FORMAT);
    }
  }

  void prepareOffscreenFramebuffer(FrameBuffer* frameBuf,
                                   VkFormat colorFormat) {
    // Color attachment
    VkImageCreateInfo imageCI = vks::initializers::imageCreateInfo();
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = colorFormat;
    imageCI.extent.width = frameBuf->width;
    imageCI.extent.height = frameBuf->height;
    imageCI.extent.depth = 1;
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    // We will sample directly from the color attachment
    imageCI.usage =
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
        vkCreateImage(device_, &imageCI, nullptr, &frameBuf->color.image));
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

    VkImageView attachments[1]{frameBuf->color.view};

    VkFramebufferCreateInfo fbufCreateInfo =
        vks::initializers::framebufferCreateInfo();
    fbufCreateInfo.renderPass = offscreenPass_.renderPass;
    fbufCreateInfo.attachmentCount = 1;
    fbufCreateInfo.pAttachments = attachments;
    fbufCreateInfo.width = frameBuf->width;
    fbufCreateInfo.height = frameBuf->height;
    fbufCreateInfo.layers = 1;

    VK_CHECK_RESULT(vkCreateFramebuffer(device_, &fbufCreateInfo, nullptr,
                                        &frameBuf->framebuffer));

    // Fill a descriptor for later use in a descriptor set
    frameBuf->descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    frameBuf->descriptor.imageView = frameBuf->color.view;
    frameBuf->descriptor.sampler = offscreenPass_.sampler;
  }

  void setupDescriptors() {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            /* descriptorCount */ MAX_CONCURRENT_FRAMES *
                /*max number of uniform buffers*/ 1),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            /* descriptorCount */ MAX_CONCURRENT_FRAMES *
                /*max number of textures*/ 2)};
    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(
            poolSizes,
            /* max number of descriptor sets that can be allocated at once*/ 1 *
                MAX_CONCURRENT_FRAMES);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo,
                                           nullptr, &descriptorPool_));

    // Layout: Advection
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0 : Fragment shader
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),
        // Binding 1 : Fragment shader field texture 1
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 1),
        // Binding 2 : Fragment shader field texture 2
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 2),
    };

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(
            setLayoutBindings.data(),
            static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.advection));

    for (auto i = 0; i < uniformBuffers_.size(); i++) {
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool_, &descriptorSetLayouts_.advection, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].advection));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].advection, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].advection.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].advection,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &velocity_field_[0].descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].advection,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 2, &velocity_field_[0].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  void preparePipelines() {
    VkPipelineLayoutCreateInfo pipelineLayoutCI =
        vks::initializers::pipelineLayoutCreateInfo(
            &descriptorSetLayouts_.advection, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.advection));

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
        vks::initializers::pipelineCreateInfo(pipelineLayouts_.advection,
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

    // Advection pipeline
    VkPipelineVertexInputStateCreateInfo emptyInputState =
        vks::initializers::pipelineVertexInputStateCreateInfo();
    pipelineCI.pVertexInputState = &emptyInputState;
    pipelineCI.layout = pipelineLayouts_.advection;
    shaderStages[0] = loadShader(getShadersPath() + "fluidsim/simple.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] =
        loadShader(getShadersPath() + "fluidsim/advection.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.advection));
  }

  // Part B (rendering)
  void render() override {
    if (!prepared_)
      return;
    VulkanExampleBase::prepareFrame();
    updateUniformBuffers();
    buildCommandBuffer();
    VulkanExampleBase::submitFrame();
  }

  // B.1
  void updateUniformBuffers() {
    ubos_.advection.viewportResolution = glm::vec2(width_, height_);
    memcpy(uniformBuffers_[currentBuffer_].advection.mapped, &ubos_.advection,
           sizeof(AdvectionUBO));
  }

  void buildCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    advectionCmd(cmdBuffer);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void advectionCmd(VkCommandBuffer& cmdBuffer) {
    VkClearValue clearValues{};
    clearValues.color = {1.0f, 0.0f, 0.0f, 1.f};

    VkRenderPassBeginInfo renderPassBeginInfo =
        vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass_;
    renderPassBeginInfo.framebuffer = frameBuffers_[currentImageIndex_];
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = width_;
    renderPassBeginInfo.renderArea.extent.height = height_;
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = &clearValues;

    vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                         VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                         VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.advection,
        0, 1, &descriptorSets_[currentBuffer_].advection, 0, nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.advection);
    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);
    drawUI(cmdBuffer);
    vkCmdEndRenderPass(cmdBuffer);
  }

  void destroyOffscreenPass() {}

  void windowResized() override {
    destroyOffscreenPass();
    vkResetDescriptorPool(device_, descriptorPool_, 0);
    setupDescriptors();
    resized_ = false;
  }

  ~VulkanExample() {
    if (device_) {
      vkDestroyPipeline(device_, pipelines_.advection, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.advection, nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.advection,
                                   nullptr);
      for (auto& buffer : uniformBuffers_) {
        buffer.advection.destroy();
      }
    }
  }
};

VULKAN_EXAMPLE_MAIN()