/*
 * Vulkan Example - Wave Simulation  *
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#include <ktx.h>
#include <ktxvulkan.h>

#include <array>
#include <vector>

#include "VulkanglTFModel.h"
#include "frustum.hpp"
#include "vulkanexamplebase.h"

// Vertex structure with position and texture coordinates
struct WaterVertex {
  float position[3];  // x, y, z
  float texCoord[2];  // u, v

  // Vulkan vertex input description
  static VkVertexInputBindingDescription getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(WaterVertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
  }

  static std::array<VkVertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
    // Position attribute(location 0) attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(WaterVertex, position);

    // Texture coordinate attribute (location 1)
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(WaterVertex, texCoord);

    return attributeDescriptions;
  }
};

struct WaterMesh {
  std::vector<WaterVertex> vertices;
  std::vector<uint32_t> indices;

  // Generate a grid mesh for water surface
  // gridSize: number of quads per side (e.g., 32 = 32x32 quads)
  // worldSize: physical size in world units
  void generateGrid(uint32_t gridSize, float worldSize) {
    vertices.clear();
    indices.clear();

    const uint32_t vertexCount = (gridSize + 1) * (gridSize + 1);
    const uint32_t quadCount = gridSize * gridSize;

    vertices.reserve(vertexCount);
    indices.reserve(quadCount * 6);  // 6 indices per quad (2 triangles)

    // Generate vertices
    for (uint32_t z = 0; z <= gridSize; ++z) {
      for (uint32_t x = 0; x <= gridSize; ++x) {
        WaterVertex vertex{};

        // Position: centered at origin, y=0 (flat plane)
        vertex.position[0] = (float(x) / gridSize - 0.5f) * worldSize;
        vertex.position[1] = 0.0f;  // Flat initially (waves applied in shader)
        vertex.position[2] = (float(z) / gridSize - 0.5f) * worldSize;

        // Texture coordinates: [0, 1] range
        vertex.texCoord[0] = float(x) / gridSize;
        vertex.texCoord[1] = float(z) / gridSize;

        vertices.push_back(vertex);
      }
    }

    // Generate indices (two triangles per quad)
    for (uint32_t z = 0; z < gridSize; ++z) {
      for (uint32_t x = 0; x < gridSize; ++x) {
        uint32_t topLeft = z * (gridSize + 1) + x;
        uint32_t topRight = topLeft + 1;
        uint32_t bottomLeft = (z + 1) * (gridSize + 1) + x;
        uint32_t bottomRight = bottomLeft + 1;

        // First triangle (top-left, bottom-left, top-right)
        indices.push_back(topLeft);
        indices.push_back(bottomLeft);
        indices.push_back(topRight);

        // Second triangle (top-right, bottom-left, bottom-right)
        indices.push_back(topRight);
        indices.push_back(bottomLeft);
        indices.push_back(bottomRight);
      }
    }
  }

  // Alternative: Generate patch list for tessellation (4 vertices per patch)
  void generatePatchGrid(uint32_t gridSize, float worldSize) {
    vertices.clear();
    indices.clear();

    const uint32_t patchCount = gridSize * gridSize;

    vertices.reserve(patchCount * 4);
    indices.reserve(patchCount * 4);

    // Generate patch vertices (each quad is a patch)
    for (uint32_t z = 0; z < gridSize; ++z) {
      for (uint32_t x = 0; x < gridSize; ++x) {
        // Four corners of the patch
        for (uint32_t corner = 0; corner < 4; ++corner) {
          WaterVertex vertex{};

          uint32_t localX = (corner == 1 || corner == 2) ? 1 : 0;
          uint32_t localZ = (corner == 2 || corner == 3) ? 1 : 0;

          float gridX = float(x + localX) / gridSize;
          float gridZ = float(z + localZ) / gridSize;

          vertex.position[0] = (gridX - 0.5f) * worldSize;
          vertex.position[1] = 0.0f;
          vertex.position[2] = (gridZ - 0.5f) * worldSize;
          vertex.texCoord[0] = gridX;
          vertex.texCoord[1] = gridZ;

          vertices.push_back(vertex);
        }

        // Indices for this patch (4 vertices)
        uint32_t baseIndex = (z * gridSize + x) * 4;
        indices.push_back(baseIndex + 0);
        indices.push_back(baseIndex + 1);
        indices.push_back(baseIndex + 2);
        indices.push_back(baseIndex + 3);
      }
    }
  }
};

class VulkanExample : public VulkanExampleBase {
 public:
  // Enable Vulkan 1.3
  VkPhysicalDeviceVulkan13Features enabledFeatures13_{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};

  struct Graphics {
    // Holds the buffers for rendering
    struct {
      vks::Buffer vertex_buffer;
      vks::Buffer index_buffer;
      uint32_t indexCount{};
    } wave_mesh_buffers;

    struct {
      vkglTF::Model sky_box{};
    } models;

    struct {
      vks::TextureCubeMap cube_map{};
    } textures;

    struct ModelViewPerspectiveUBO {
      glm::mat4 perspective;
      glm::mat4 view;
    };
    struct {
      ModelViewPerspectiveUBO sky_box;
    } ubos;

    struct UniformBuffers {
      vks::Buffer sky_box;
    };
    std::array<UniformBuffers, maxConcurrentFrames> uniform_buffers;

    struct Pipelines {
      VkPipeline sky_box{VK_NULL_HANDLE};
    } pipelines;

    struct {
      VkDescriptorSetLayout sky_box{VK_NULL_HANDLE};
    } descriptor_set_layouts;

    struct {
      VkPipelineLayout sky_box{VK_NULL_HANDLE};
    } pipeline_layouts;

    struct DescriptorSets {
      VkDescriptorSet sky_box{VK_NULL_HANDLE};
    };
    std::array<DescriptorSets, maxConcurrentFrames> descriptor_sets;
  } graphics_;

  void setupDescriptors() {
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              maxConcurrentFrames * 1),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            maxConcurrentFrames * 1)};
    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(poolSizes,
                                                    maxConcurrentFrames * 1);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr,
                                           &descriptorPool));

    // Layouts
    VkDescriptorSetLayoutCreateInfo descriptorLayout;
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;

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
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr,
                                    &graphics_.descriptor_set_layouts.sky_box));

    for (auto i = 0; i < graphics_.uniform_buffers.size(); i++) {
      // Skybox
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool, &graphics_.descriptor_set_layouts.sky_box, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device, &allocInfo, &graphics_.descriptor_sets[i].sky_box));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              graphics_.descriptor_sets[i].sky_box,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
              &graphics_.uniform_buffers[i].sky_box.descriptor),
          vks::initializers::writeDescriptorSet(
              graphics_.descriptor_sets[i].sky_box,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
              &graphics_.textures.cube_map.descriptor),
      };
      vkUpdateDescriptorSets(device,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  void preparePipelines() {
    // Layouts
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;

    // Skybox
    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &graphics_.descriptor_set_layouts.sky_box, 1);
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr,
                               &graphics_.pipeline_layouts.sky_box));

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

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
        vks::initializers::pipelineInputAssemblyStateCreateInfo(
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);

    VkGraphicsPipelineCreateInfo pipelineCI =
        vks::initializers::pipelineCreateInfo();
    pipelineCI.layout = graphics_.pipeline_layouts.sky_box;
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

    // Skybox (cubemap) pipeline
    depthStencilState.depthWriteEnable = VK_FALSE;
    depthStencilState.depthTestEnable = VK_TRUE;
    rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;
    rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
    // Revert to triangle list topology
    // Reset tessellation state
    pipelineCI.pTessellationState = nullptr;
    pipelineCI.stageCount = 2;
    shaderStages[0] = loadShader(getShadersPath() + "wave/skybox.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "wave/skybox.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1,
                                              &pipelineCI, nullptr,
                                              &graphics_.pipelines.sky_box));
  }

  // Prepare and initialize uniform buffer containing shader uniforms
  void prepareUniformBuffers() {
    for (auto& buffer : graphics_.uniform_buffers) {
      // Skybox vertex shader uniform buffer
      VK_CHECK_RESULT(vulkanDevice->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.sky_box, sizeof(graphics_.ubos.sky_box)));
      VK_CHECK_RESULT(buffer.sky_box.map());
    }
  }

  void updateUniformBuffers() {
    graphics_.ubos.sky_box.perspective = camera.matrices.perspective;
    graphics_.ubos.sky_box.view = glm::mat4(glm::mat3(camera.matrices.view));
    memcpy(graphics_.uniform_buffers[currentBuffer].sky_box.mapped,
           &graphics_.ubos.sky_box, sizeof(Graphics::ModelViewPerspectiveUBO));
  }

  void setupWaterMesh() {
    // 1. Generate the mesh
    WaterMesh waterMesh;
    waterMesh.generatePatchGrid(16, 100.0f);  // 16x16 patches

    // 2. Create Vulkan buffers
    uint32_t vertex_bufferSize =
        waterMesh.vertices.size() * sizeof(WaterVertex);
    uint32_t index_bufferSize = waterMesh.indices.size() * sizeof(uint32_t);
    vks::Buffer vertexStaging, indexStaging;

    // 3. Store for rendering
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &vertexStaging, vertex_bufferSize, waterMesh.vertices.data()));

    VK_CHECK_RESULT(vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &indexStaging, index_bufferSize, waterMesh.indices.data()));

    VK_CHECK_RESULT(vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &graphics_.wave_mesh_buffers.vertex_buffer, vertex_bufferSize));

    VK_CHECK_RESULT(vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &graphics_.wave_mesh_buffers.index_buffer, index_bufferSize));

    // Copy from staging buffers
    VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    VkBufferCopy copyRegion = {};

    copyRegion.size = vertex_bufferSize;
    vkCmdCopyBuffer(copyCmd, vertexStaging.buffer,
                    graphics_.wave_mesh_buffers.vertex_buffer.buffer, 1,
                    &copyRegion);

    copyRegion.size = index_bufferSize;
    vkCmdCopyBuffer(copyCmd, indexStaging.buffer,
                    graphics_.wave_mesh_buffers.index_buffer.buffer, 1,
                    &copyRegion);

    vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

    vkDestroyBuffer(device, vertexStaging.buffer, nullptr);
    vkFreeMemory(device, vertexStaging.memory, nullptr);
    vkDestroyBuffer(device, indexStaging.buffer, nullptr);
    vkFreeMemory(device, indexStaging.memory, nullptr);
  }

  void prepare() override {
    VulkanExampleBase::prepare();
    loadAssets();
    setupWaterMesh();
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
                            graphics_.pipeline_layouts.sky_box, 0, 1,
                            &graphics_.descriptor_sets[currentBuffer].sky_box,
                            0, nullptr);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics_.pipelines.sky_box);
    graphics_.models.sky_box.draw(cmdBuffer);

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
    }
  }

  void loadAssets() {
    VkSamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();

    // Skybox cube model
    const uint32_t glTFLoadingFlags =
        vkglTF::FileLoadingFlags::PreTransformVertices |
        vkglTF::FileLoadingFlags::PreMultiplyVertexColors |
        vkglTF::FileLoadingFlags::FlipY;
    graphics_.models.sky_box.loadFromFile(getAssetPath() + "models/cube.gltf",
                                          vulkanDevice, queue,
                                          glTFLoadingFlags);
    // Skybox textures
    graphics_.textures.cube_map.loadFromFile(
        getExamplesBasePath() + "wave/cartoon_skybox.ktx",
        VK_FORMAT_R8G8B8A8_SRGB, vulkanDevice, queue);
  }

  VulkanExample() : VulkanExampleBase() {
    title = "Wave Simulation";
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
      vkDestroyPipeline(device, graphics_.pipelines.sky_box, nullptr);
    }
  }
};

VULKAN_EXAMPLE_MAIN()
