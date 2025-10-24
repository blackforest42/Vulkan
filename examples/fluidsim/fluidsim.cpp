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

class VulkanExample : public VulkanExampleBase {
 public:
  struct UBO {
    glm::vec2 viewportResolution;
  };
  struct {
    UBO simple;
  } ubos_;

  struct UniformBuffers {
    vks::Buffer simple;
  };
  std::array<UniformBuffers, MAX_CONCURRENT_FRAMES> uniformBuffers_{};

  std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSets_{};
  VkDescriptorSetLayout descriptorSetLayout_{};
  VkPipeline pipeline_{};
  VkPipelineLayout pipelineLayout_{};

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
          &buffer.simple, sizeof(UBO), &ubos_.simple));
      VK_CHECK_RESULT(buffer.simple.map());
    }
  }

  void setupDescriptors() {
    std::vector<VkDescriptorPoolSize> poolSizes = {};
    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(poolSizes,
                                                    MAX_CONCURRENT_FRAMES);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo,
                                           nullptr, &descriptorPool_));

    // Layout:
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0 : Fragment shader
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0)};

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(
            setLayoutBindings.data(),
            static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device_, &descriptorSetLayoutCI, nullptr, &descriptorSetLayout_));

    for (auto i = 0; i < uniformBuffers_.size(); i++) {
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool_, &descriptorSetLayout_, 1);
      VK_CHECK_RESULT(
          vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSets_[i]));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].simple.descriptor)};
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  void preparePipelines() {
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

    VkPipelineVertexInputStateCreateInfo emptyInputState =
        vks::initializers::pipelineVertexInputStateCreateInfo();
    pipelineCI.pVertexInputState = &emptyInputState;
    pipelineCI.layout = pipelineLayout_;
    shaderStages[0] = loadShader(getShadersPath() + "fluidsim/simple.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "fluidsim/simple.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipeline_));
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
    ubos_.simple.viewportResolution = glm::vec2(width_, height_);
    memcpy(uniformBuffers_[currentBuffer_].simple.mapped, &ubos_.simple,
           sizeof(UBO));
  }

  void buildCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    VkClearValue clearValues{};
    clearValues.color = {0.0f, 0.0f, 0.0f, 0.f};

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

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout_, 0, 1,
                            &descriptorSets_[currentBuffer_], 0, nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);
    drawUI(cmdBuffer);
    vkCmdEndRenderPass(cmdBuffer);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
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
      vkDestroyPipeline(device_, pipeline_, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
      for (auto& buffer : uniformBuffers_) {
        buffer.simple.destroy();
      }
    }
  }
};

VULKAN_EXAMPLE_MAIN()