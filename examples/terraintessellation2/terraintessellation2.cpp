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
  // Enable Vulkan 1.3
  VkPhysicalDeviceVulkan13Features enabledFeatures13_{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};

  bool wireframe = false;

  struct {
    vkglTF::Model skyBox{};
  } models_;

  struct {
    vks::TextureCubeMap cubeMap{};
    vks::Texture2D heightMap{};
  } textures_;

  struct ModelViewPerspectiveUBO {
    glm::mat4 perspective;
    glm::mat4 view;
  };
  struct {
    ModelViewPerspectiveUBO terrain, skyBox;
  } ubos_;

  struct UniformBuffers {
    vks::Buffer terrain;
    vks::Buffer skybox;
  };
  std::array<UniformBuffers, maxConcurrentFrames> uniformBuffers_;

  // Holds the buffers for rendering the tessellated terrain
  struct {
    vks::Buffer vertexBuffer;
    vks::Buffer indexBuffer;
    uint32_t indexCount{};
  } terrain_;

  struct Pipelines {
    VkPipeline terrain{VK_NULL_HANDLE};
    VkPipeline wireframe{VK_NULL_HANDLE};
    VkPipeline skyBox{VK_NULL_HANDLE};
  } pipelines_;

  struct {
    VkDescriptorSetLayout terrain{VK_NULL_HANDLE};
    VkDescriptorSetLayout skyBox{VK_NULL_HANDLE};
  } descriptorSetLayouts_;

  struct {
    VkPipelineLayout terrain{VK_NULL_HANDLE};
    VkPipelineLayout skyBox{VK_NULL_HANDLE};
  } pipelineLayouts_;

  struct DescriptorSets {
    VkDescriptorSet terrain{VK_NULL_HANDLE};
    VkDescriptorSet skyBox{VK_NULL_HANDLE};
  };
  std::array<DescriptorSets, maxConcurrentFrames> descriptorSets_;

  // Generate a terrain quad patch with normals based on heightmap data
  void generateTerrain() {
    std::string filename =
        getExamplesBasePath() + "terraintessellation2/iceland_heightmap_r8.ktx";

    ktxTexture* ktxTexture;
    ktxResult result = ktxTexture_CreateFromNamedFile(
        filename.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
    assert(result == KTX_SUCCESS);
    ktx_size_t ktxSize = ktxTexture_GetImageSize(ktxTexture, 0);
    ktx_uint8_t* ktxImage = ktxTexture_GetData(ktxTexture);
    const uint32_t TEXTURE_WIDTH = ktxTexture->baseWidth;
    const uint32_t TEXTURE_HEIGHT = ktxTexture->baseHeight;
    ktxTexture_Destroy(ktxTexture);

    const uint32_t n_patches{128};
    const float patch_width = (float)TEXTURE_WIDTH / n_patches;
    const float patch_height = (float)TEXTURE_HEIGHT / n_patches;
    // We use the Vertex definition from the glTF model loader, so we can re-use
    // the vertex input state
    constexpr uint32_t vertexCount = n_patches * n_patches;
    std::vector<vkglTF::Vertex> vertices(vertexCount);

    // Generate a two-dimensional vertex patch
    for (auto row = 0; row < n_patches; row++) {
      for (auto col = 0; col < n_patches; col++) {
        uint32_t index = (row + col * n_patches);
        vertices[index].pos[0] = row * patch_width + patch_width / 2.0f -
                                 (float)n_patches * patch_width / 2.0f;
        vertices[index].pos[1] = 0.0f;
        vertices[index].pos[2] = col * patch_height + patch_height / 2.0f -
                                 (float)n_patches * patch_height / 2.0f;
        vertices[index].uv =
            glm::vec2((float)row / (n_patches), (float)col / (n_patches));
      }
    }

    // Generate indices
    constexpr uint32_t w = (n_patches - 1);
    terrain_.indexCount = w * w * 4;
    std::vector<uint32_t> indices(terrain_.indexCount);
    for (auto x = 0; x < w; x++) {
      for (auto y = 0; y < w; y++) {
        uint32_t index = (x + y * w) * 4;
        indices[index] = (x + y * n_patches);
        indices[index + 1] = indices[index] + n_patches;
        indices[index + 2] = indices[index + 1] + 1;
        indices[index + 3] = indices[index] + 1;
      }
    }

    terrain_.indexCount = (uint32_t)indices.size();

    // Allocate buffer space for vertices and indices
    size_t vertexBufferSize = vertices.size() * sizeof(vkglTF::Vertex);
    size_t indexBufferSize = indices.size() * sizeof(uint32_t);
    vks::Buffer vertexBuffer, indexBuffer;

    // Staging Buffer (Source)
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &vertexBuffer, vertexBufferSize, vertices.data()));
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &indexBuffer, indexBufferSize, indices.data()));

    // GPU Device Buffer (Destination)
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &terrain_.vertexBuffer,
        vertexBufferSize));
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &terrain_.indexBuffer,
        indexBufferSize));

    // Copy from staging buffers
    VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    VkBufferCopy copyRegion = {};

    copyRegion.size = vertexBufferSize;
    vkCmdCopyBuffer(copyCmd, vertexBuffer.buffer, terrain_.vertexBuffer.buffer,
                    1, &copyRegion);

    copyRegion.size = indexBufferSize;
    vkCmdCopyBuffer(copyCmd, indexBuffer.buffer, terrain_.indexBuffer.buffer, 1,
                    &copyRegion);

    vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

    vkDestroyBuffer(device, vertexBuffer.buffer, nullptr);
    vkFreeMemory(device, vertexBuffer.memory, nullptr);
    vkDestroyBuffer(device, indexBuffer.buffer, nullptr);
    vkFreeMemory(device, indexBuffer.memory, nullptr);
  }

  void setupDescriptors() {
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              maxConcurrentFrames * 2),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            maxConcurrentFrames * 2)};
    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(poolSizes,
                                                    maxConcurrentFrames * 2);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr,
                                           &descriptorPool));

    // Layouts
    VkDescriptorSetLayoutCreateInfo descriptorLayout;
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;

    // Terrain
    setLayoutBindings = {
        // Binding 0 : Shared Tessellation shader UBO
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
            0),
        // Binding 1 : Height map
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
                VK_SHADER_STAGE_FRAGMENT_BIT,
            1),
    };
    descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device, &descriptorLayout, nullptr, &descriptorSetLayouts_.terrain));

    // Skybox
    setLayoutBindings = {
        // Binding 0 : Vertex shader ubo
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
            /*binding id*/ 0),
        // Binding 1 : Color map
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, /*binding id*/ 1),
    };
    descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device, &descriptorLayout, nullptr, &descriptorSetLayouts_.skyBox));

    for (auto i = 0; i < uniformBuffers_.size(); i++) {
      // Terrain
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool, &descriptorSetLayouts_.terrain, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo,
                                               &descriptorSets_[i].terrain));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          // Binding 0 : MVP (model x view x perspective) mat4
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].terrain, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
              &uniformBuffers_[i].terrain.descriptor),
          // Binding 1 : Height map
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].terrain,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
              &textures_.heightMap.descriptor),
      };
      vkUpdateDescriptorSets(device,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Skybox
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool, &descriptorSetLayouts_.skyBox, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo,
                                               &descriptorSets_[i].skyBox));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].skyBox, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
              &uniformBuffers_[i].skybox.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].skyBox,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
              &textures_.cubeMap.descriptor),
      };
      vkUpdateDescriptorSets(device,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  void preparePipelines() {
    // Layouts
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;

    // Terrain
    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.terrain, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo,
                                           nullptr, &pipelineLayouts_.terrain));
    // Skybox
    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.skyBox, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo,
                                           nullptr, &pipelineLayouts_.skyBox));

    // Pipeline
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
    std::array<VkPipelineShaderStageCreateInfo, 4> shaderStages{};

    // Tessellation stage input is a patch of quads
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
        vks::initializers::pipelineInputAssemblyStateCreateInfo(
            VK_PRIMITIVE_TOPOLOGY_PATCH_LIST, 0, VK_FALSE);
    VkPipelineTessellationStateCreateInfo tessellationState =
        vks::initializers::pipelineTessellationStateCreateInfo(4);

    // Terrain tessellation pipeline stages
    shaderStages[0] =
        loadShader(getShadersPath() + "terraintessellation2/terrain.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] =
        loadShader(getShadersPath() + "terraintessellation2/terrain.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    shaderStages[2] =
        loadShader(getShadersPath() + "terraintessellation2/terrain.tesc.spv",
                   VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
    shaderStages[3] =
        loadShader(getShadersPath() + "terraintessellation2/terrain.tese.spv",
                   VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);

    VkGraphicsPipelineCreateInfo pipelineCI =
        vks::initializers::pipelineCreateInfo();
    pipelineCI.layout = pipelineLayouts_.terrain;
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
    pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState(
        {vkglTF::VertexComponent::Position, vkglTF::VertexComponent::UV});

    // Dynamic rendering. New create info to define color, depth and stencil
    // attachments at pipeline create time
    VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo{};
    pipelineRenderingCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    pipelineRenderingCreateInfo.colorAttachmentCount = 1;
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &swapChain.colorFormat;
    pipelineRenderingCreateInfo.depthAttachmentFormat = depthFormat;
    pipelineRenderingCreateInfo.stencilAttachmentFormat = depthFormat;
    // Chain into the pipeline create info
    pipelineCI.pNext = &pipelineRenderingCreateInfo;

    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines_.terrain));

    // Wireframe
    if (deviceFeatures.fillModeNonSolid) {
      rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
      VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1,
                                                &pipelineCI, nullptr,
                                                &pipelines_.wireframe));
    }

    // Skybox (cubemap) pipeline
    depthStencilState.depthWriteEnable = VK_FALSE;
    depthStencilState.depthTestEnable = VK_TRUE;
    rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;
    rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
    // Revert to triangle list topology
    inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    // Reset tessellation state
    pipelineCI.pTessellationState = nullptr;
    pipelineCI.layout = pipelineLayouts_.skyBox;
    pipelineCI.stageCount = 2;
    shaderStages[0] =
        loadShader(getShadersPath() + "terraintessellation2/skybox.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] =
        loadShader(getShadersPath() + "terraintessellation2/skybox.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines_.skyBox));
  }

  // Prepare and initialize uniform buffer containing shader uniforms
  void prepareUniformBuffers() {
    for (auto& buffer : uniformBuffers_) {
      // Terrain vertex shader uniform buffer
      VK_CHECK_RESULT(vulkanDevice->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.terrain, sizeof(ModelViewPerspectiveUBO)));
      VK_CHECK_RESULT(buffer.terrain.map());

      // Skybox vertex shader uniform buffer
      VK_CHECK_RESULT(
          vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                     &buffer.skybox, sizeof(ubos_.skyBox)));
      VK_CHECK_RESULT(buffer.skybox.map());
    }
  }

  void updateUniformBuffers() {
    ubos_.terrain.perspective = camera.matrices.perspective;
    ubos_.terrain.view = camera.matrices.view;
    memcpy(uniformBuffers_[currentBuffer].terrain.mapped, &ubos_.terrain,
           sizeof(ModelViewPerspectiveUBO));

    ubos_.skyBox.perspective = camera.matrices.perspective;
    ubos_.skyBox.view = glm::mat4(glm::mat3(camera.matrices.view));
    memcpy(uniformBuffers_[currentBuffer].skybox.mapped, &ubos_.skyBox,
           sizeof(ModelViewPerspectiveUBO));
  }

  void prepare() override {
    VulkanExampleBase::prepare();
    loadAssets();
    generateTerrain();
    prepareUniformBuffers();
    setupDescriptors();
    preparePipelines();
    prepared = true;
  }

  void setupRenderPass() override {
    // With VK_KHR_dynamic_rendering we no longer need a render pass, so
    // skip the sample base render pass setup
    renderPass = VK_NULL_HANDLE;
  }

  void setupFrameBuffer() override {
    // With VK_KHR_dynamic_rendering we no longer need a frame buffer
    // LEAVE THIS EMPTY
  }

  void buildCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers[currentBuffer];

    const VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    // With dynamic rendering there are no subpass dependencies, so we need to
    // take care of proper layout transitions by using barriers This set of
    // barriers prepares the color and depth images for output
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, swapChain.images[currentImageIndex], 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, depthStencil.image, 0,
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
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachment.imageView = swapChain.imageViews[currentImageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {0.0f, 0.0f, 1.0f, 0.0f};

    // A single depth stencil attachment info can be used, but they can also be
    // specified separately. When both are specified separately, the only
    // requirement is that the image view is identical.
    VkRenderingAttachmentInfo depthStencilAttachment{};
    depthStencilAttachment.sType =
        VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    depthStencilAttachment.imageView = depthStencil.view;
    depthStencilAttachment.imageLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthStencilAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthStencilAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthStencilAttachment.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    renderingInfo.renderArea = {0, 0, width, height};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthStencilAttachment;
    renderingInfo.pStencilAttachment = &depthStencilAttachment;

    vkCmdBeginRendering(cmdBuffer, &renderingInfo);

    VkViewport viewport =
        vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    // Skybox
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayouts_.skyBox, 0, 1,
                            &descriptorSets_[currentBuffer].skyBox, 0, nullptr);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.skyBox);
    models_.skyBox.draw(cmdBuffer);

    // Terrain/wireframe
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      wireframe ? pipelines_.wireframe : pipelines_.terrain);
    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.terrain, 0,
        1, &descriptorSets_[currentBuffer].terrain, 0, nullptr);
    VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &terrain_.vertexBuffer.buffer,
                           offsets);
    vkCmdBindIndexBuffer(cmdBuffer, terrain_.indexBuffer.buffer, 0,
                         VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmdBuffer, terrain_.indexCount, 1, 0, 0, 0);

    drawUI(cmdBuffer);

    // End dynamic rendering
    vkCmdEndRendering(cmdBuffer);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void render() override {
    if (!prepared) {
      return;
    }
    VulkanExampleBase::prepareFrame();
    updateUniformBuffers();
    buildCommandBuffer();
    VulkanExampleBase::submitFrame();
  }

  void OnUpdateUIOverlay(vks::UIOverlay* overlay) override {
    if (deviceFeatures.fillModeNonSolid) {
      overlay->checkBox("Wireframe", &wireframe);
    }
  }

  void loadAssets() {
    // Height data is stored in a one-channel texture
    textures_.heightMap.loadFromFile(
        getExamplesBasePath() + "terraintessellation2/iceland_heightmap_r8.ktx",
        VK_FORMAT_R8_UNORM, vulkanDevice, queue);

    VkSamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();

    // Skybox cube model
    const uint32_t glTFLoadingFlags =
        vkglTF::FileLoadingFlags::PreTransformVertices |
        vkglTF::FileLoadingFlags::PreMultiplyVertexColors |
        vkglTF::FileLoadingFlags::FlipY;
    models_.skyBox.loadFromFile(getAssetPath() + "models/cube.gltf",
                                vulkanDevice, queue, glTFLoadingFlags);
    // Skybox textures
    textures_.cubeMap.loadFromFile(
        getExamplesBasePath() + "terraintessellation2/cartoon_skybox.ktx",
        VK_FORMAT_R8G8B8A8_SRGB, vulkanDevice, queue);
  }

  VulkanExample() : VulkanExampleBase() {
    title = "Dynamic terrain tessellation 2";
    camera.type_ = Camera::CameraType::firstperson;
    camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 1024.0f);
    camera.setTranslation(glm::vec3(18.0f, 64.5f, 57.5f));
    camera.movementSpeed = 100.0f;

    apiVersion = VK_API_VERSION_1_3;

    // Dynamic rendering
    enabledFeatures13_.dynamicRendering = VK_TRUE;

    deviceCreatepNextChain = &enabledFeatures13_;
  }

  ~VulkanExample() override {
    if (device) {
      vkDestroyPipeline(device, pipelines_.terrain, nullptr);
      if (pipelines_.wireframe != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipelines_.wireframe, nullptr);
      }
      textures_.heightMap.destroy();
      textures_.cubeMap.destroy();
      for (auto& buffer : uniformBuffers_) {
        buffer.terrain.destroy();
      }
    }
  }
};

VULKAN_EXAMPLE_MAIN()
