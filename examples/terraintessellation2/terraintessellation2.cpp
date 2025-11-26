/*
 * Vulkan Example - Dynamic terrain tessellation part 2
 *
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#include <ktx.h>
#include <ktxvulkan.h>

#include "VulkanglTFModel.h"
#include "frustum.hpp"
#include "vulkanexamplebase.h"

class VulkanExample : public VulkanExampleBase {
 public:
  struct {
    vks::TextureCubeMap skyBox{};
    vks::Texture2D heightMap{};
  } textures_;

  struct TerrainUBO {
    glm::mat4 mvp;
  } terrain_ubo;

  struct UniformBuffers {
    vks::Buffer terrain;
  };
  std::array<UniformBuffers, MAX_CONCURRENT_FRAMES> uniformBuffers_;

  // Holds the buffers for rendering the tessellated terrain
  struct {
    vks::Buffer vertexBuffer;
    vks::Buffer indexBuffer;
    uint32_t indexCount{};
  } terrain_;

  struct Pipelines {
    VkPipeline terrain{VK_NULL_HANDLE};
  } pipelines_;

  struct {
    VkDescriptorSetLayout terrain{VK_NULL_HANDLE};
  } descriptorSetLayouts_;

  struct {
    VkPipelineLayout terrain{VK_NULL_HANDLE};
  } pipelineLayouts_;

  struct DescriptorSets {
    VkDescriptorSet terrain{VK_NULL_HANDLE};
  };
  std::array<DescriptorSets, MAX_CONCURRENT_FRAMES> descriptorSets_;

  VulkanExample() : VulkanExampleBase() {
    title = "Dynamic terrain tessellation 2";
    camera_.type_ = Camera::CameraType::firstperson;
    camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 512.0f);
    camera_.setRotation(glm::vec3(-12.0f, 159.0f, 0.0f));
    camera_.setTranslation(glm::vec3(18.0f, 22.5f, 57.5f));
    camera_.movementSpeed = 10.0f;
  }

  // Generate a terrain quad patch with normals based on heightmap data
  void generateTerrain() {
    std::string filename = getAssetPath() + "textures/iceland_heightmap.ktx";

    ktxResult result;
    ktxTexture* ktxTexture;
    result = ktxTexture_CreateFromNamedFile(
        filename.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
    assert(result == KTX_SUCCESS);
    ktx_size_t ktxSize = ktxTexture_GetImageSize(ktxTexture, 0);
    ktx_uint8_t* ktxImage = ktxTexture_GetData(ktxTexture);
    uint32_t COLS = ktxTexture->baseWidth;
    uint32_t ROWS = ktxTexture->baseHeight;
    float height_scale = 64.0f / 256.0f;
    // Adjusts vertical translation of height map. e.g. Below or above
    // surface.
    float height_shift = 16.f;
    // Generate vertices
    std::vector<vkglTF::Vertex> vertices(ROWS * COLS);
    int i = 0;
    for (int r = 0; r < ROWS; r++) {
      for (int c = 0; c < COLS; c++) {
        // Read height map for 'y' value. Normalize to [0, 1] then rescale and
        // translate (up or down).
        float height = ktxImage[i] * height_scale - height_shift;

        vertices[i].pos[0] = c - COLS / 2.f;
        vertices[i].pos[1] = height;
        vertices[i].pos[2] = r - ROWS / 2.f;

        i++;
      }
    }
    ktxTexture_Destroy(ktxTexture);

    // Generate indices
    std::vector<int> indices;
    for (int r = 0; r < ROWS - 1; r++) {
      for (int c = 0; c < COLS; c++) {
        for (int k = 0; k < 2; k++) {
          indices.push_back(c + COLS * (r + k));
        }
      }
    }
    terrain_.indexCount = indices.size();

    // Allocate buffer space for vertices and indices
    uint32_t vertexBufferSize = vertices.size() * sizeof(vkglTF::Vertex);
    uint32_t indexBufferSize = indices.size() * sizeof(uint32_t);
    vks::Buffer vertexBuffer, indexBuffer;

    // Source
    VK_CHECK_RESULT(vulkanDevice_->createBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &vertexBuffer, vertexBufferSize, vertices.data()));
    VK_CHECK_RESULT(vulkanDevice_->createBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &indexBuffer, indexBufferSize, indices.data()));

    // Destination
    VK_CHECK_RESULT(vulkanDevice_->createBuffer(
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &terrain_.vertexBuffer,
        vertexBufferSize));
    VK_CHECK_RESULT(vulkanDevice_->createBuffer(
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &terrain_.indexBuffer,
        indexBufferSize));

    // Copy from staging buffers
    VkCommandBuffer copyCmd = vulkanDevice_->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    VkBufferCopy copyRegion = {};

    copyRegion.size = vertexBufferSize;
    vkCmdCopyBuffer(copyCmd, vertexBuffer.buffer, terrain_.vertexBuffer.buffer,
                    1, &copyRegion);

    copyRegion.size = indexBufferSize;
    vkCmdCopyBuffer(copyCmd, indexBuffer.buffer, terrain_.indexBuffer.buffer, 1,
                    &copyRegion);

    vulkanDevice_->flushCommandBuffer(copyCmd, queue_, true);

    vkDestroyBuffer(device_, vertexBuffer.buffer, nullptr);
    vkFreeMemory(device_, vertexBuffer.memory, nullptr);
    vkDestroyBuffer(device_, indexBuffer.buffer, nullptr);
    vkFreeMemory(device_, indexBuffer.memory, nullptr);
  }

  void setupDescriptors() {
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              MAX_CONCURRENT_FRAMES * 1),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            MAX_CONCURRENT_FRAMES * 1)};
    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(poolSizes,
                                                    MAX_CONCURRENT_FRAMES * 1);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo,
                                           nullptr, &descriptorPool_));

    // Layouts
    VkDescriptorSetLayoutCreateInfo descriptorLayout;
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;

    // Terrain
    setLayoutBindings = {
        // Binding 0 : Vertex shader ubo
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
    };
    descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device_, &descriptorLayout, nullptr, &descriptorSetLayouts_.terrain));

    for (auto i = 0; i < uniformBuffers_.size(); i++) {
      // Terrain
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool_, &descriptorSetLayouts_.terrain, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].terrain));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          // Binding 0 : model, view, projection mat4
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].terrain, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
              &uniformBuffers_[i].terrain.descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  void preparePipelines() {
    // Layouts
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;

    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.terrain, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo,
                                           nullptr, &pipelineLayouts_.terrain));

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
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState =
        vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

    VkGraphicsPipelineCreateInfo pipelineCI =
        vks::initializers::pipelineCreateInfo(pipelineLayouts_.terrain,
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

    // Terrain tessellation pipeline
    shaderStages[0] =
        loadShader(getShadersPath() + "terraintessellation2/mesh.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] =
        loadShader(getShadersPath() + "terraintessellation2/mesh.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);

    pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState(
        {vkglTF::VertexComponent::Position});
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.terrain));
  }

  // Prepare and initialize uniform buffer containing shader uniforms
  void prepareUniformBuffers() {
    for (auto& buffer : uniformBuffers_) {
      // Skysphere vertex shader uniform buffer
      VK_CHECK_RESULT(
          vulkanDevice_->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      &buffer.terrain, sizeof(TerrainUBO)));
      VK_CHECK_RESULT(buffer.terrain.map());
    }
  }

  void updateUniformBuffers() {
    // Vertex shader
    terrain_ubo.mvp = camera_.matrices_.perspective *
                      glm::mat4(glm::mat3(camera_.matrices_.view));
    memcpy(uniformBuffers_[currentBuffer_].terrain.mapped, &terrain_ubo,
           sizeof(TerrainUBO));
  }

  void prepare() {
    VulkanExampleBase::prepare();
    loadAssets();
    generateTerrain();
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
    clearValues[0].color = {0, 0, 0, 1};
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

    vkCmdSetLineWidth(cmdBuffer, 1.0f);

    VkDeviceSize offsets[1] = {0};

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.terrain);
    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.terrain, 0,
        1, &descriptorSets_[currentBuffer_].terrain, 0, nullptr);
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &terrain_.vertexBuffer.buffer,
                           offsets);
    vkCmdBindIndexBuffer(cmdBuffer, terrain_.indexBuffer.buffer, 0,
                         VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmdBuffer, terrain_.indexCount, 1, 0, 0, 0);

    drawUI(cmdBuffer);

    vkCmdEndRenderPass(cmdBuffer);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void render() override {
    if (!prepared_) {
      return;
    }
    VulkanExampleBase::prepareFrame();
    updateUniformBuffers();
    buildCommandBuffer();
    VulkanExampleBase::submitFrame();
  }

  virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay) {}

  // Enable physical device features required for this example
  virtual void getEnabledFeatures() {
    // Tessellation shader support is required for this example
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
    };
  }

  void loadAssets() {
    textures_.skyBox.loadFromFile(
        getAssetPath() + "textures/cartoon_sky_cubemap.ktx",
        VK_FORMAT_R8G8B8A8_SRGB, vulkanDevice_, queue_);

    // Height data is stored in a one-channel texture
    textures_.heightMap.loadFromFile(
        getAssetPath() + "textures/iceland_heightmap.ktx", VK_FORMAT_R8_SRGB,
        vulkanDevice_, queue_);

    VkSamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();

    // Setup a mirroring sampler for the height map
    vkDestroySampler(device_, textures_.heightMap.sampler, nullptr);
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    samplerInfo.addressModeV = samplerInfo.addressModeU;
    samplerInfo.addressModeW = samplerInfo.addressModeU;
    samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = (float)textures_.heightMap.mipLevels;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(vkCreateSampler(device_, &samplerInfo, nullptr,
                                    &textures_.heightMap.sampler));
    textures_.heightMap.descriptor.sampler = textures_.heightMap.sampler;
  }

  ~VulkanExample() {
    if (device_) {
      textures_.heightMap.destroy();
      textures_.skyBox.destroy();
      for (auto& buffer : uniformBuffers_) {
        buffer.terrain.destroy();
      }
    }
  }
};

VULKAN_EXAMPLE_MAIN()
