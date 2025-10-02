/*
* Vulkan Example - Using inline uniform blocks for passing data to shader stages
at descriptor setup

* Note: Requires a device that supports the VK_EXT_inline_uniform_block
extension
*
* Relevant code parts are marked with [POI]
*
* Copyright (C) 2018-2025 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT)
(http://opensource.org/licenses/MIT)
*/

#include "VulkanglTFModel.h"
#include "vulkanexamplebase.h"

class VulkanExample : public VulkanExampleBase {
 public:
  VkPhysicalDeviceInlineUniformBlockFeaturesEXT
      enabledInlineUniformBlockFeatures{};

  vkglTF::Model model;

  struct Object {
    struct Material {
      float roughness;
      float metallic;
      float r, g, b;
      float ambient;
    } material;
    std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSets{};
    void setRandomMaterial(bool applyRandomSeed) {
      std::random_device rndDevice;
      std::default_random_engine rndEngine(applyRandomSeed ? rndDevice() : 0);
      std::uniform_real_distribution<float> rndDist(0.1f, 1.0f);
      material.r = rndDist(rndEngine);
      material.g = rndDist(rndEngine);
      material.b = rndDist(rndEngine);
      material.ambient = 0.0025f;
      material.roughness = glm::clamp(rndDist(rndEngine), 0.005f, 1.0f);
      material.metallic = glm::clamp(rndDist(rndEngine), 0.005f, 1.0f);
    }
  };
  std::array<Object, 16> objects{};

  struct UniformData {
    glm::mat4 projection;
    glm::mat4 model;
    glm::mat4 view;
    glm::vec3 camPos;
  } uniformData_;
  std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers_;

  VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
  VkPipeline pipeline{VK_NULL_HANDLE};

  struct DescriptorSetLaysts {
    VkDescriptorSetLayout scene{VK_NULL_HANDLE};
    VkDescriptorSetLayout object{VK_NULL_HANDLE};
  } descriptorSetLayouts_;
  std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSets_{};

  bool doUpdateMaterials{false};

  VulkanExample() : VulkanExampleBase() {
    title = "Inline uniform blocks";
    camera_.type_ = Camera::CameraType::firstperson;
    camera_.setPosition(glm::vec3(0.0f, 0.0f, -10.0f));
    camera_.setRotation(glm::vec3(0.0, 0.0f, 0.0f));
    camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);
    camera_.movementSpeed = 4.0f;
    camera_.rotationSpeed = 0.25f;

    /*
            [POI] Enable extensions required for inline uniform blocks
    */
    enabledDeviceExtensions_.push_back(
        VK_EXT_INLINE_UNIFORM_BLOCK_EXTENSION_NAME);
    enabledDeviceExtensions_.push_back(VK_KHR_MAINTENANCE1_EXTENSION_NAME);
    enabledInstanceExtensions_.push_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    /*
            [POI] We also need to enable the inline uniform block feature (using
       the dedicated physical device structure)
    */
    enabledInlineUniformBlockFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES_EXT;
    enabledInlineUniformBlockFeatures.inlineUniformBlock = VK_TRUE;
    deviceCreatepNextChain_ = &enabledInlineUniformBlockFeatures;
  }

  ~VulkanExample() {
    if (device_) {
      vkDestroyPipeline(device_, pipeline, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.scene,
                                   nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.object,
                                   nullptr);
      for (auto& buffer : uniformBuffers_) {
        buffer.destroy();
      }
    }
  }

  void loadAssets() {
    model.loadFromFile(getAssetPath() + "models/sphere.gltf", vulkanDevice_,
                       queue_);

    // Setup random materials for every object in the scene
    for (uint32_t i = 0; i < objects.size(); i++) {
      objects[i].setRandomMaterial(!benchmark.active);
    }
  }

  void setupDescriptors() {
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              MAX_CONCURRENT_FRAMES),
        /* [POI] Allocate inline uniform blocks */
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT,
            static_cast<uint32_t>(objects.size()) * sizeof(Object::Material)),
    };
    VkDescriptorPoolCreateInfo descriptorPoolCI =
        vks::initializers::descriptorPoolCreateInfo(
            poolSizes, (static_cast<uint32_t>(objects.size()) + 1) *
                           MAX_CONCURRENT_FRAMES);
    /*
            [POI] New structure that has to be chained into the descriptor
       pool's createinfo if you want to allocate inline uniform blocks
    */
    VkDescriptorPoolInlineUniformBlockCreateInfoEXT
        descriptorPoolInlineUniformBlockCreateInfo{};
    descriptorPoolInlineUniformBlockCreateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO_EXT;
    descriptorPoolInlineUniformBlockCreateInfo.maxInlineUniformBlockBindings =
        static_cast<uint32_t>(objects.size());
    descriptorPoolCI.pNext = &descriptorPoolInlineUniformBlockCreateInfo;

    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolCI, nullptr,
                                           &descriptorPool_));

    // Layouts
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};
    VkDescriptorSetLayoutCreateInfo descriptorLayoutCI{};
    // Scene matrices
    setLayoutBindings = {
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0),
    };
    descriptorLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device_, &descriptorLayoutCI, nullptr, &descriptorSetLayouts_.scene));
    setLayoutBindings = {
        /*
                [POI] Setup inline uniform block for set 1 at binding 0 (see
           fragment shader) Descriptor count for an inline uniform block
           contains data sizes of the block (last parameter)
        */
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT,
            VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Object::Material)),
    };
    descriptorLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device_, &descriptorLayoutCI, nullptr, &descriptorSetLayouts_.object));

    // Sets per frame, just like the buffers themselves
    for (auto i = 0; i < uniformBuffers_.size(); i++) {
      VkDescriptorSetAllocateInfo descriptorAllocateInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool_, &descriptorSetLayouts_.scene, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &descriptorAllocateInfo,
                                               &descriptorSets_[i]));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
              &uniformBuffers_[i].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
      // Objects with inline uniform blocks
      for (auto& object : objects) {
        VkDescriptorSetAllocateInfo descriptorAllocateInfo =
            vks::initializers::descriptorSetAllocateInfo(
                descriptorPool_, &descriptorSetLayouts_.object, 1);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(
            device_, &descriptorAllocateInfo, &object.descriptorSets[i]));

        /*
                [POI] New structure that defines size and data of the inline
           uniform block needs to be chained into the write descriptor set We
           will be using this inline uniform block to pass per-object material
           information to the fragment shader
        */
        VkWriteDescriptorSetInlineUniformBlockEXT
            writeDescriptorSetInlineUniformBlock{};
        writeDescriptorSetInlineUniformBlock.sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK_EXT;
        writeDescriptorSetInlineUniformBlock.dataSize =
            sizeof(Object::Material);
        // Uniform data for the inline block
        writeDescriptorSetInlineUniformBlock.pData = &object.material;

        /*
                [POI] Setup the inline uniform block
        */
        VkWriteDescriptorSet writeDescriptorSet{};
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.descriptorType =
            VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT;
        writeDescriptorSet.dstSet = object.descriptorSets[i];
        writeDescriptorSet.dstBinding = 0;
        // Descriptor count for an inline uniform block contains data sizes of
        // the block(last parameter)
        writeDescriptorSet.descriptorCount = sizeof(Object::Material);
        // Chain inline uniform block structure
        writeDescriptorSet.pNext = &writeDescriptorSetInlineUniformBlock;

        vkUpdateDescriptorSets(device_, 1, &writeDescriptorSet, 0, nullptr);
      }
    }
  }

  void preparePipelines() {
    /*
            [POI] Pipeline layout usin two sets, one for the scene matrices and
       one for the per-object inline uniform blocks
    */
    std::vector<VkDescriptorSetLayout> setLayouts = {
        descriptorSetLayouts_.scene,  // Set 0 = Scene matrices
        descriptorSetLayouts_.object  // Set 1 = Object inline uniform block
    };
    VkPipelineLayoutCreateInfo pipelineLayoutCI =
        vks::initializers::pipelineLayoutCreateInfo(
            setLayouts.data(), static_cast<uint32_t>(setLayouts.size()));

    // We use push constants for passing object positions
    std::vector<VkPushConstantRange> pushConstantRanges = {
        vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT,
                                             sizeof(glm::vec3), 0),
    };
    pipelineLayoutCI.pushConstantRangeCount = 1;
    pipelineLayoutCI.pPushConstantRanges = pushConstantRanges.data();

    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayout));

    // Pipeline
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI =
        vks::initializers::pipelineInputAssemblyStateCreateInfo(
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
    VkPipelineRasterizationStateCreateInfo rasterizationStateCI =
        vks::initializers::pipelineRasterizationStateCreateInfo(
            VK_POLYGON_MODE_FILL, VK_CULL_MODE_FRONT_BIT,
            VK_FRONT_FACE_COUNTER_CLOCKWISE);
    VkPipelineColorBlendAttachmentState blendAttachmentState =
        vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
    VkPipelineColorBlendStateCreateInfo colorBlendStateCI =
        vks::initializers::pipelineColorBlendStateCreateInfo(
            1, &blendAttachmentState);
    VkPipelineDepthStencilStateCreateInfo depthStencilStateCI =
        vks::initializers::pipelineDepthStencilStateCreateInfo(
            VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
    VkPipelineViewportStateCreateInfo viewportStateCI =
        vks::initializers::pipelineViewportStateCreateInfo(1, 1);
    VkPipelineMultisampleStateCreateInfo multisampleStateCI =
        vks::initializers::pipelineMultisampleStateCreateInfo(
            VK_SAMPLE_COUNT_1_BIT);
    std::vector<VkDynamicState> dynamicStateEnables = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicStateCI =
        vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

    VkGraphicsPipelineCreateInfo pipelineCI =
        vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass_);
    pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
    pipelineCI.pRasterizationState = &rasterizationStateCI;
    pipelineCI.pColorBlendState = &colorBlendStateCI;
    pipelineCI.pMultisampleState = &multisampleStateCI;
    pipelineCI.pViewportState = &viewportStateCI;
    pipelineCI.pDepthStencilState = &depthStencilStateCI;
    pipelineCI.pDynamicState = &dynamicStateCI;
    pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCI.pStages = shaderStages.data();
    pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState(
        {vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal});

    shaderStages[0] =
        loadShader(getShadersPath() + "inlineuniformblocks/pbr.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] =
        loadShader(getShadersPath() + "inlineuniformblocks/pbr.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr, &pipeline));
  }

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
    uniformData_.projection = camera_.matrices_.perspective;
    uniformData_.view = camera_.matrices_.view;
    uniformData_.model = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));
    uniformData_.camPos = camera_.position_ * glm::vec3(-1.0f, 1.0f, -1.0f);
    memcpy(uniformBuffers_[currentBuffer_].mapped, &uniformData_,
           sizeof(UniformData));
  }

  /*
          [POI] Update descriptor set data at runtime using inline uniform
     blocks
  */
  void updateMaterials() {
    for (auto& object : objects) {
      /*
              [POI] New structure that defines size and data of the inline
         uniform block needs to be chained into the write descriptor set We will
         be using this inline uniform block to pass per-object material
         information to the fragment shader
      */
      VkWriteDescriptorSetInlineUniformBlockEXT
          writeDescriptorSetInlineUniformBlock{};
      writeDescriptorSetInlineUniformBlock.sType =
          VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK_EXT;
      writeDescriptorSetInlineUniformBlock.dataSize = sizeof(Object::Material);
      // Uniform data for the inline block
      writeDescriptorSetInlineUniformBlock.pData = &object.material;

      /*
              [POI] Update the object's inline uniform block
      */
      VkWriteDescriptorSet writeDescriptorSet{};
      writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writeDescriptorSet.descriptorType =
          VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT;
      writeDescriptorSet.dstSet = object.descriptorSets[currentBuffer_];
      writeDescriptorSet.dstBinding = 0;
      writeDescriptorSet.descriptorCount = sizeof(Object::Material);
      writeDescriptorSet.pNext = &writeDescriptorSetInlineUniformBlock;

      vkUpdateDescriptorSets(device_, 1, &writeDescriptorSet, 0, nullptr);
    }
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
    clearValues[0].color = {{0.15f, 0.15f, 0.15f, 1.0f}};
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

    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    // Render objects
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    uint32_t objcount = static_cast<uint32_t>(objects.size());
    for (uint32_t i = 0; i < objcount; i++) {
      /*
              [POI] Bind descriptor sets
              Set 0 = Scene matrices:
              Set 1 = Object inline uniform block (In shader pbr.frag: layout
         (set = 1, binding = 0) uniform UniformInline ... )
      */
      std::vector<VkDescriptorSet> sets = {
          descriptorSets_[currentBuffer_],
          objects[i].descriptorSets[currentBuffer_]};
      vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipelineLayout, 0, 2, sets.data(), 0, nullptr);

      glm::vec3 pos =
          glm::vec3(sin(glm::radians(i * (360.0f / objcount))),
                    cos(glm::radians(i * (360.0f / objcount))), 0.0f) *
          3.5f;

      vkCmdPushConstants(cmdBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                         0, sizeof(glm::vec3), &pos);
      model.draw(cmdBuffer);
    }
    drawUI(cmdBuffer);

    vkCmdEndRenderPass(cmdBuffer);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  virtual void render() {
    if (!prepared_)
      return;
    VulkanExampleBase::prepareFrame();
    updateUniformBuffers();
    updateMaterials();
    buildCommandBuffer();
    VulkanExampleBase::submitFrame();
  }

  virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay) {
    if (overlay->button("Randomize")) {
      // Randomize material properties
      for (uint32_t i = 0; i < objects.size(); i++) {
        objects[i].setRandomMaterial(!benchmark.active);
      }
    }
  }
};

VULKAN_EXAMPLE_MAIN()
