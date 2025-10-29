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
  const uint32_t JACOBI_ITERATIONS = 1;

  struct AdvectionUBO {
    glm::vec2 bufferResolution{};
    float timestep{.1f};
  };

  struct BoundaryUBO {
    glm::vec2 bufferResolution{};
    float scale{1.f};
  };

  struct JacobiUBO {
    float alpha{1.f};
    float beta{0.25f};
  };

  struct {
    AdvectionUBO advection;
    BoundaryUBO boundary;
    JacobiUBO jacobi;
  } ubos_;

  struct UniformBuffers {
    vks::Buffer advection;
    vks::Buffer boundary;
    vks::Buffer jacobi;
  };
  std::array<UniformBuffers, MAX_CONCURRENT_FRAMES> uniformBuffers_{};

  struct {
    VkDescriptorSetLayout advection;
    VkDescriptorSetLayout boundary;
    VkDescriptorSetLayout jacobi;
  } descriptorSetLayouts_{};

  struct DescriptorSets {
    VkDescriptorSet advection;
    VkDescriptorSet boundaryVelocity;
    VkDescriptorSet boundaryPressure;
    VkDescriptorSet jacobiVelocity;
    VkDescriptorSet jacobiPressure;
  };
  std::array<DescriptorSets, MAX_CONCURRENT_FRAMES> descriptorSets_{};

  VkPipelineLayout pipelineLayout_{};

  struct {
    VkPipeline advection;
    VkPipeline boundary;
    VkPipeline jacobi;
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
  FrameBuffer divergence_{};

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
      // Advection
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.advection, sizeof(AdvectionUBO), &ubos_.advection));
      VK_CHECK_RESULT(buffer.advection.map());

      // Boundary
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.boundary, sizeof(BoundaryUBO), &ubos_.boundary));
      VK_CHECK_RESULT(buffer.boundary.map());

      // Jacobi
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.jacobi, sizeof(JacobiUBO), &ubos_.jacobi));
      VK_CHECK_RESULT(buffer.jacobi.map());
    }
  }

  void prepareOffscreen() {
    // Create a separate render pass for the offscreen rendering as it may
    // differ from the one used for scene rendering
    VkAttachmentDescription attchmentDescriptions;
    // Color attachment
    attchmentDescriptions.format = FB_COLOR_FORMAT;
    attchmentDescriptions.samples = VK_SAMPLE_COUNT_1_BIT;
    attchmentDescriptions.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attchmentDescriptions.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attchmentDescriptions.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attchmentDescriptions.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attchmentDescriptions.initialLayout =
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
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

    // Velocity field
    for (auto& fb : velocity_field_) {
      fb.width = width_;
      fb.height = height_;
      prepareOffscreenFramebuffer(&fb, FB_COLOR_FORMAT);
    }

    // Pressure field
    for (auto& fb : pressure_field_) {
      fb.width = width_;
      fb.height = height_;
      prepareOffscreenFramebuffer(&fb, FB_COLOR_FORMAT);
    }

    // Divergence field
    divergence_.width = width_;
    divergence_.height = height_;
    prepareOffscreenFramebuffer(&divergence_, FB_COLOR_FORMAT);
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
            /* max number of descriptor sets that can be allocated at once*/ 5 *
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

    // Layout: Boundary
    setLayoutBindings = {
        // Binding 0 : Fragment shader
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),
        // Binding 1 : Fragment shader field texture 1
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 1)};
    descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.boundary));

    // Layout: Jacobi
    setLayoutBindings = {
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
    descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI,
                                                nullptr,
                                                &descriptorSetLayouts_.jacobi));

    for (auto i = 0; i < uniformBuffers_.size(); i++) {
      // Advection
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

      // Boundary: Velocty
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.boundary, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &descriptorSets_[i].boundaryVelocity));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].boundaryVelocity,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].boundary.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].boundaryVelocity,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &velocity_field_[0].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Boundary: Pressure
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.boundary, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &descriptorSets_[i].boundaryPressure));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].boundaryPressure,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].boundary.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].boundaryPressure,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &pressure_field_[0].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Jacobi: Velocity
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.jacobi, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &descriptorSets_[i].jacobiVelocity));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].jacobiVelocity,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].jacobi.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].jacobiVelocity,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &velocity_field_[0].descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].jacobiVelocity,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 2, &velocity_field_[0].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Jacobi: Pressure
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.jacobi, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &descriptorSets_[i].jacobiPressure));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].jacobiPressure,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].jacobi.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].jacobiPressure,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &pressure_field_[0].descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].jacobiPressure,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 2, &divergence_.descriptor),
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
                                           &pipelineLayout_));

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

    // Advection pipeline
    VkPipelineVertexInputStateCreateInfo emptyInputState =
        vks::initializers::pipelineVertexInputStateCreateInfo();
    pipelineCI.pVertexInputState = &emptyInputState;
    pipelineCI.layout = pipelineLayout_;
    shaderStages[0] = loadShader(getShadersPath() + "fluidsim/simple.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] =
        loadShader(getShadersPath() + "fluidsim/advection.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.advection));

    // Boundary pipeline
    pipelineCI.layout = pipelineLayout_;
    shaderStages[1] =
        loadShader(getShadersPath() + "fluidsim/boundary.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.boundary));

    // Jacobi pipeline
    pipelineCI.layout = pipelineLayout_;
    shaderStages[1] = loadShader(getShadersPath() + "fluidsim/jacobi.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.jacobi));
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
    ubos_.advection.bufferResolution = glm::vec2(width_, height_);
    memcpy(uniformBuffers_[currentBuffer_].advection.mapped, &ubos_.advection,
           sizeof(AdvectionUBO));
    ubos_.boundary.bufferResolution = glm::vec2(width_, height_);
    memcpy(uniformBuffers_[currentBuffer_].boundary.mapped, &ubos_.boundary,
           sizeof(BoundaryUBO));
  }

  void buildCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    // Advection
    velocityBoundaryCmd(cmdBuffer);
    advectionCmd(cmdBuffer);

    // Jacobi
    for (int i = 0; i < JACOBI_ITERATIONS; i++) {
      // velocityBoundaryCmd(cmdBuffer);
      // velocityJacobiCmd(cmdBuffer);
    }

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void advectionCmd(VkCommandBuffer& cmdBuffer) {
    VkClearValue clearValues{};
    clearValues.color = {1.0f, 0.0f, 0.0f, 1.f};

    VkRenderPassBeginInfo renderPassBeginInfo =
        vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = offscreenPass_.renderPass;
    renderPassBeginInfo.framebuffer = velocity_field_[1].framebuffer;
    renderPassBeginInfo.renderArea.offset.x = 1;
    renderPassBeginInfo.renderArea.offset.y = 1;
    renderPassBeginInfo.renderArea.extent.width = width_ - 2;
    renderPassBeginInfo.renderArea.extent.height = height_ - 2;
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
        cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1,
        &descriptorSets_[currentBuffer_].advection, 0, nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.advection);
    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

    // IMPORTANT: This barrier is to serialize WRITES BEFORE READS
    {
      VkMemoryBarrier memBarrier = {};
      memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0,
                           nullptr, 0, nullptr);
    }

    vkCmdEndRenderPass(cmdBuffer);

    copyImage(cmdBuffer, velocity_field_[1].color.image,
              velocity_field_[0].color.image);
  }

  void velocityBoundaryCmd(VkCommandBuffer& cmdBuffer) {
    ubos_.boundary.scale = -1;
    memcpy(uniformBuffers_[currentBuffer_].boundary.mapped, &ubos_.boundary,
           sizeof(BoundaryUBO));
    boundaryCmd(cmdBuffer, velocity_field_,
                &descriptorSets_[currentBuffer_].boundaryVelocity);
  }

  void pressureBoundaryCmd(VkCommandBuffer& cmdBuffer) {
    ubos_.boundary.scale = 1;
    memcpy(uniformBuffers_[currentBuffer_].boundary.mapped, &ubos_.boundary,
           sizeof(BoundaryUBO));
    boundaryCmd(cmdBuffer, pressure_field_,
                &descriptorSets_[currentBuffer_].boundaryPressure);
  }

  void boundaryCmd(VkCommandBuffer& cmdBuffer,
                   std::array<FrameBuffer, 2>& output_field,
                   VkDescriptorSet* descriptor_set) {
    VkClearValue clearValues{};
    clearValues.color = {0.0f, 0.0f, 0.0f, 1.f};

    VkRenderPassBeginInfo renderPassBeginInfo =
        vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = offscreenPass_.renderPass;
    renderPassBeginInfo.framebuffer = output_field[1].framebuffer;
    // renderPassBeginInfo.renderPass = renderPass_;
    // renderPassBeginInfo.framebuffer = frameBuffers_[currentImageIndex_];
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

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout_, 0, 1, descriptor_set, 0, nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.boundary);
    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

    // IMPORTANT: This barrier is to serialize WRITES BEFORE READS
    {
      VkMemoryBarrier memBarrier = {};
      memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0,
                           nullptr, 0, nullptr);
    }
    vkCmdEndRenderPass(cmdBuffer);

    // copyImage(cmdBuffer, output_field[1].color.image,
    //           output_field[0].color.image);
  }

  void velocityJacobiCmd(VkCommandBuffer& cmdBuffer) {
    jacobiCmd(cmdBuffer, velocity_field_,
              &descriptorSets_[currentBuffer_].jacobiVelocity);
  }

  void pressureJacobiCmd(VkCommandBuffer& cmdBuffer) {
    jacobiCmd(cmdBuffer, pressure_field_,
              &descriptorSets_[currentBuffer_].jacobiPressure);
  }

  void jacobiCmd(VkCommandBuffer& cmdBuffer,
                 std::array<FrameBuffer, 2>& output_field,
                 VkDescriptorSet* descriptor_set) {
    VkClearValue clearValues{};
    clearValues.color = {0.0f, 0.0f, 0.0f, 1.f};

    VkRenderPassBeginInfo renderPassBeginInfo =
        vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = offscreenPass_.renderPass;
    renderPassBeginInfo.framebuffer = output_field[1].framebuffer;
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

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout_, 0, 1, descriptor_set, 0, nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.jacobi);
    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

    // IMPORTANT: This barrier is to serialize WRITES BEFORE READS
    {
      VkMemoryBarrier memBarrier = {};
      memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0,
                           nullptr, 0, nullptr);
    }
    vkCmdEndRenderPass(cmdBuffer);

    copyImage(cmdBuffer, output_field[1].color.image,
              output_field[0].color.image);
  }

  // Copy framebuffer color attachmebt from source to dest
  void copyImage(VkCommandBuffer& cmdBuffer, VkImage& source, VkImage& dest) {
    VkImageCopy copyRegion = {};

    copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.srcSubresource.mipLevel = 0;
    copyRegion.srcSubresource.baseArrayLayer = 0;
    copyRegion.srcSubresource.layerCount = 1;
    copyRegion.srcOffset = {0, 0, 0};

    copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.dstSubresource.mipLevel = 0;
    copyRegion.dstSubresource.baseArrayLayer = 0;
    copyRegion.dstSubresource.layerCount = 1;
    copyRegion.dstOffset = {0, 0, 0};

    copyRegion.extent.width = static_cast<uint32_t>(width_);
    copyRegion.extent.height = static_cast<uint32_t>(height_);
    copyRegion.extent.depth = 1;

    // Copy output of write to read buffer
    vkCmdCopyImage(cmdBuffer, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dest, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
  }

  void windowResized() override {
    destroyOffscreenPass();
    prepareOffscreen();
    vkResetDescriptorPool(device_, descriptorPool_, 0);
    setupDescriptors();
    resized_ = false;
  }

  void destroyOffscreenPass() {
    vkDestroyRenderPass(device_, offscreenPass_.renderPass, nullptr);
    for (FrameBuffer fb : velocity_field_) {
      vkDestroyFramebuffer(device_, fb.framebuffer, nullptr);
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

  ~VulkanExample() {
    if (device_) {
      vkDestroyPipeline(device_, pipelines_.advection, nullptr);
      vkDestroyPipeline(device_, pipelines_.boundary, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.advection,
                                   nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.boundary,
                                   nullptr);
      for (auto& buffer : uniformBuffers_) {
        buffer.advection.destroy();
        buffer.boundary.destroy();
      }
    }
  }
};

VULKAN_EXAMPLE_MAIN()