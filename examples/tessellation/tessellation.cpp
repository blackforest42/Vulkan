/*
 * Vulkan Example - Tessellation shader PN triangles
 *
 * Based on http://alex.vlachos.com/graphics/CurvedPNTriangles.pdf
 * Shaders based on
 * http://onrendering.blogspot.de/2011/12/tessellation-on-gpu-curved-pn-triangles.html
 *
 * Copyright (C) 2016-2025 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#include "VulkanglTFModel.h"
#include "vulkanexamplebase.h"

class VulkanExample : public VulkanExampleBase {
 public:
  bool splitScreen = true;
  bool wireframe = true;

  vkglTF::Model model;

  // One uniform data block is used by both tessellation shader stages
  struct UniformData {
    glm::mat4 projection;
    glm::mat4 modelView;
    float tessAlpha = 1.0f;
    float tessLevel = 3.0f;
  } uniformData_;
  std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers_;

  struct Pipelines {
    VkPipeline solid{VK_NULL_HANDLE};
    VkPipeline wire{VK_NULL_HANDLE};
    VkPipeline solidPassThrough{VK_NULL_HANDLE};
    VkPipeline wirePassThrough{VK_NULL_HANDLE};
  } pipelines_;

  VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
  VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
  std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSets_{};

  VulkanExample() : VulkanExampleBase() {
    title_ = "Tessellation shader (PN Triangles)";
    camera_.type_ = Camera::CameraType::lookat;
    camera_.setPosition(glm::vec3(0.0f, 0.0f, -4.0f));
    camera_.setRotation(glm::vec3(-350.0f, 60.0f, 0.0f));
    camera_.setPerspective(
        45.0f, (float)(width_ * ((splitScreen) ? 0.5f : 1.0f)) / (float)height_,
        0.1f, 256.0f);
  }

  ~VulkanExample() {
    if (device_) {
      // Clean up used Vulkan resources
      // Note : Inherited destructor cleans up resources stored in base class
      vkDestroyPipeline(device_, pipelines_.solid, nullptr);
      if (pipelines_.wire != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipelines_.wire, nullptr);
      };
      vkDestroyPipeline(device_, pipelines_.solidPassThrough, nullptr);
      if (pipelines_.wirePassThrough != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipelines_.wirePassThrough, nullptr);
      };
      vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayout, nullptr);
      for (auto& buffer : uniformBuffers_) {
        buffer.destroy();
      }
    }
  }

  // Enable physical device features required for this example
  virtual void getEnabledFeatures() {
    // Example requires tessellation shaders
    if (deviceFeatures_.tessellationShader) {
      enabledFeatures_.tessellationShader = VK_TRUE;
    } else {
      vks::tools::exitFatal(
          "Selected GPU does not support tessellation shaders!",
          VK_ERROR_FEATURE_NOT_PRESENT);
    }
    // Fill mode non solid is required for wireframe display
    if (deviceFeatures_.fillModeNonSolid) {
      enabledFeatures_.fillModeNonSolid = VK_TRUE;
    } else {
      wireframe = false;
    }
    if (deviceFeatures_.samplerAnisotropy) {
      enabledFeatures_.samplerAnisotropy = VK_TRUE;
    }
  }

  void loadAssets() {
    model.loadFromFile(getAssetPath() + "models/deer.gltf", vulkanDevice_,
                       queue_,
                       vkglTF::FileLoadingFlags::PreTransformVertices |
                           vkglTF::FileLoadingFlags::FlipY);
  }

  void setupDescriptors() {
    // Pool
    const std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              MAX_CONCURRENT_FRAMES),
    };
    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(poolSizes,
                                                    MAX_CONCURRENT_FRAMES);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo,
                                           nullptr, &descriptorPool_));

    // Layout
    const std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0 : Tessellation shader ubo
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
            0),
    };
    VkDescriptorSetLayoutCreateInfo descriptorLayout =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorLayout,
                                                nullptr, &descriptorSetLayout));

    // Sets per frame, just like the buffers themselves
    VkDescriptorSetAllocateInfo allocInfo =
        vks::initializers::descriptorSetAllocateInfo(descriptorPool_,
                                                     &descriptorSetLayout, 1);
    for (auto i = 0; i < uniformBuffers_.size(); i++) {
      VK_CHECK_RESULT(
          vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSets_[i]));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
              &uniformBuffers_[i].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  void preparePipelines() {
    // Layout uses set 0 for passing tessellation shader ubos and set 1 for
    // fragment shader images (taken from glTF model)
    const std::vector<VkDescriptorSetLayout> setLayouts = {
        descriptorSetLayout,
        vkglTF::descriptorSetLayoutImage,
    };
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
        vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), 2);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo,
                                           nullptr, &pipelineLayout));

    // Pipelines
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
        vks::initializers::pipelineInputAssemblyStateCreateInfo(
            VK_PRIMITIVE_TOPOLOGY_PATCH_LIST, 0, VK_FALSE);
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
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_LINE_WIDTH};
    VkPipelineDynamicStateCreateInfo dynamicState =
        vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables,
                                                          0);
    VkPipelineTessellationStateCreateInfo tessellationState =
        vks::initializers::pipelineTessellationStateCreateInfo(3);
    std::array<VkPipelineShaderStageCreateInfo, 4> shaderStages;

    // Tessellation pipelines
    shaderStages[0] =
        loadShader(getShadersPath() + "tessellation/base.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] =
        loadShader(getShadersPath() + "tessellation/base.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    shaderStages[2] =
        loadShader(getShadersPath() + "tessellation/pntriangles.tesc.spv",
                   VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
    shaderStages[3] =
        loadShader(getShadersPath() + "tessellation/pntriangles.tese.spv",
                   VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);

    VkGraphicsPipelineCreateInfo pipelineCI =
        vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass_, 0);
    pipelineCI.pInputAssemblyState = &inputAssemblyState;
    pipelineCI.pRasterizationState = &rasterizationState;
    pipelineCI.pColorBlendState = &colorBlendState;
    pipelineCI.pMultisampleState = &multisampleState;
    pipelineCI.pViewportState = &viewportState;
    pipelineCI.pDepthStencilState = &depthStencilState;
    pipelineCI.pDynamicState = &dynamicState;
    pipelineCI.pTessellationState = &tessellationState;
    pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCI.pStages = shaderStages.data();
    pipelineCI.renderPass = renderPass_;
    pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState(
        {vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal,
         vkglTF::VertexComponent::UV});

    // Tessellation pipelines
    // Solid
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.solid));
    // Wireframe
    if (deviceFeatures_.fillModeNonSolid) {
      rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
      VK_CHECK_RESULT(vkCreateGraphicsPipelines(
          device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.wire));
    }

    // Pass through pipelines
    // Load pass through tessellation shaders (Vert and frag are reused)
    shaderStages[2] =
        loadShader(getShadersPath() + "tessellation/passthrough.tesc.spv",
                   VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
    shaderStages[3] =
        loadShader(getShadersPath() + "tessellation/passthrough.tese.spv",
                   VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);

    // Solid
    rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.solidPassThrough));
    // Wireframe
    if (deviceFeatures_.fillModeNonSolid) {
      rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
      VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                                &pipelineCI, nullptr,
                                                &pipelines_.wirePassThrough));
    }
  }

  // Prepare and initialize uniform buffer containing shader uniforms
  void prepareUniformBuffers() {
    for (auto& buffer : uniformBuffers_) {
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer, sizeof(UniformData), &uniformData_));
      VK_CHECK_RESULT(buffer.map());
    }
  }

  void updateUniformBuffers() {
    // Adjust camera perspective if split screen is enabled
    camera_.setPerspective(
        45.0f, (float)(width_ * ((splitScreen) ? 0.5f : 1.0f)) / (float)height_,
        0.1f, 256.0f);
    uniformData_.projection = camera_.matrices_.perspective;
    uniformData_.modelView = camera_.matrices_.view;
    memcpy(uniformBuffers_[currentBuffer_].mapped, &uniformData_,
           sizeof(UniformData));
  }

  void prepare() {
    VulkanExampleBase::prepare();
    loadAssets();
    prepareUniformBuffers();
    setupDescriptors();
    preparePipelines();
    prepared_ = true;
  }

  void buildCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();

    VkClearValue clearValues[2]{};
    clearValues[0].color = {{0.5f, 0.5f, 0.5f, 0.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassBeginInfo =
        vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass_;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = width_;
    renderPassBeginInfo.renderArea.extent.height = height_;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;
    renderPassBeginInfo.framebuffer = frameBuffers_[currentImageIndex_];

    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                         VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = vks::initializers::viewport(
        splitScreen ? (float)width_ / 2.0f : (float)width_, (float)height_,
        0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdSetLineWidth(cmdBuffer, 1.0f);

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 0, 1,
                            &descriptorSets_[currentBuffer_], 0, nullptr);

    if (splitScreen) {
      vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);
      vkCmdBindPipeline(
          cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
          wireframe ? pipelines_.wirePassThrough : pipelines_.solidPassThrough);
      model.draw(cmdBuffer, vkglTF::RenderFlags::BindImages, pipelineLayout);
      viewport.x = float(width_) / 2;
    }

    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      wireframe ? pipelines_.wire : pipelines_.solid);
    model.draw(cmdBuffer, vkglTF::RenderFlags::BindImages, pipelineLayout);

    drawUI(cmdBuffer);

    vkCmdEndRenderPass(cmdBuffer);

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
      overlay->inputFloat("Tessellation level", &uniformData_.tessLevel, 0.25f,
                          2);
      if (deviceFeatures_.fillModeNonSolid) {
        overlay->checkBox("Wireframe", &wireframe);
        if (overlay->checkBox("Splitscreen", &splitScreen)) {
          camera_.setPerspective(
              45.0f,
              (float)(width_ * ((splitScreen) ? 0.5f : 1.0f)) / (float)height_,
              0.1f, 256.0f);
        }
      }
    }
  }
};

VULKAN_EXAMPLE_MAIN()
