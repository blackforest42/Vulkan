/*
 * Vulkan Example - Wave Simulation  *
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#include <ktxvulkan.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

#include "VulkanglTFModel.h"
#include "frustum.hpp"
#include "vulkanexamplebase.h"

constexpr uint32_t COMPUTE_TEXTURE_DIMENSION = 512;
static_assert((COMPUTE_TEXTURE_DIMENSION & (COMPUTE_TEXTURE_DIMENSION - 1)) ==
                  0,
              "COMPUTE_TEXTURE_DIMENSION must be power of two for FFT");

// Vertex structure with position and texture coordinates
struct WaterVertex {
  glm::vec3 position;  // x, y, z
  glm::vec2 uv;        // u, v

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
    attributeDescriptions[1].offset = offsetof(WaterVertex, uv);

    return attributeDescriptions;
  }
};

struct WaterMesh {
  std::vector<WaterVertex> vertices;
  std::vector<uint32_t> indices;

  void generateUnitQuad() {
    vertices = {
        // Position (x, y, z)              // UV (u, v)
        {{-1.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},  // Bottom-Left
        {{1.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},   // Bottom-Right
        {{1.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},    // Top-Right
        {{-1.0f, 0.0f, 1.0f}, {0.0f, 1.0f}}    // Top-Left
    };
    indices = {0, 1, 2, 2, 3, 0};
  }
};

struct OceanParams {
  glm::vec4 wind_dir_speed_amp;  // xy: wind dir, z: wind speed, w: amplitude
  glm::vec4 time_patch_chop_height;  // x: time, y: patch length, z: chop, w:
                                     // height scale
  glm::ivec4 grid;                   // x: N, y: logN
};

struct UiFeatures {
  bool pause_wave{false};
  bool show_mesh_only{false};
  float patch_scale{512.f};
  float time_step{1};
  float sun_scale{100.0f};
} ui_features;

class VulkanExample : public VulkanExampleBase {
 public:
  // Enable Vulkan 1.3
  VkPhysicalDeviceVulkan13Features enabledFeatures13_{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
  };

  // Dynamic state feature
  VkPhysicalDeviceExtendedDynamicState3FeaturesEXT dynamicState3Features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};

  // Command to enable dynamic states during rendering
  PFN_vkCmdSetPolygonModeEXT vkCmdSetPolygonModeEXT{VK_NULL_HANDLE};

  // Time tracking for simulation
  std::chrono::steady_clock::time_point last_sim_time_{};
  bool sim_time_initialized_{false};

  // Debug labeling ext
  static constexpr std::array<float, 4> debugColor_ = {.7f, 0.4f, 0.4f, 1.0f};
  PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT{nullptr};
  PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT{nullptr};

  // Handles all compute pipelines
  struct Compute {
    static constexpr int WORKGROUP_SIZE = 16;

    // Used to check if compute and graphics queue
    // families differ and require additional barriers
    uint32_t queueFamilyIndex{0};
    // Separate queue for compute commands (queue family may
    // differ from the one used for graphics)
    VkQueue queue{};
    // Use a separate command pool (queue family may
    // differ from the one used for graphics)
    VkCommandPool commandPool{};
    // Command buffer storing the dispatch commands and
    // barriers
    std::array<VkCommandBuffer, maxConcurrentFrames> commandBuffers{};
    // Fences to make sure command buffers are done
    std::array<VkFence, maxConcurrentFrames> fences{};

    // Semaphores for submission ordering
    struct ComputeSemaphores {
      VkSemaphore ready{VK_NULL_HANDLE};
      VkSemaphore complete{VK_NULL_HANDLE};
    };
    std::array<ComputeSemaphores, maxConcurrentFrames> semaphores{};

    // Contains all Vulkan objects that are required to store and use a 2D tex
    struct Texture2D {
      VkSampler sampler = VK_NULL_HANDLE;
      VkImage image = VK_NULL_HANDLE;
      VkImageLayout imageLayout{};
      VkDeviceMemory deviceMemory = VK_NULL_HANDLE;
      VkImageView view = VK_NULL_HANDLE;
      VkDescriptorImageInfo descriptor{};
      VkFormat format{};
      uint32_t width{0};
      uint32_t height{0};
      uint32_t depth{0};
      uint32_t mipLevels{0};
    };

    Texture2D spectrum_h0;
    Texture2D spectrum_ht_ping;
    Texture2D spectrum_ht_pong;
    Texture2D height_map;
    Texture2D normal_map;

    struct {
      OceanParams ocean;
    } ubos;

    struct UniformBuffers {
      vks::Buffer ocean;
    };

    // Buffers
    std::array<UniformBuffers, maxConcurrentFrames> uniform_buffers;

    // Pipelines
    struct {
      VkPipeline init_spectrum{VK_NULL_HANDLE};
      VkPipeline spectrum{VK_NULL_HANDLE};
      VkPipeline fft{VK_NULL_HANDLE};
      VkPipeline resolve_height{VK_NULL_HANDLE};
      VkPipeline normals{VK_NULL_HANDLE};
    } pipelines{};

    // Pipeline Layout
    struct {
      VkPipelineLayout ocean{VK_NULL_HANDLE};
    } pipeline_layouts;

    // Descriptor Layout
    struct {
      VkDescriptorSetLayout ocean{VK_NULL_HANDLE};
    } descriptor_set_layouts;

    // Descriptor Sets
    struct DescriptorSets {
      VkDescriptorSet ocean{VK_NULL_HANDLE};
    };
    std::array<DescriptorSets, maxConcurrentFrames> descriptor_sets{};
    bool spectrum_initialized{false};
  } compute_;

  // Handles graphics rendering pipelines
  struct Graphics {
    // families differ and require additional barriers
    uint32_t queueFamilyIndex{0};

    struct {
      vks::Buffer vertex_buffer;
      vks::Buffer index_buffer;
      uint32_t index_count{};
    } wave_mesh_buffers;

    struct {
      vkglTF::Model sky_box{};
      vkglTF::Model sun{};
    } models;

    struct {
      vks::TextureCubeMap cube_map{};
    } textures;

    struct SkyBoxUBO {
      glm::mat4 perspective;
      glm::mat4 view;
    };

    struct WaveUBO {
      alignas(16) glm::mat4 perspective;
      alignas(16) glm::mat4 view;
      alignas(16) glm::vec3 camera_position;
      alignas(8) glm::vec2 screen_res;
      alignas(8) float patch_scale{};
      alignas(16) glm::vec3 sun_position;
      alignas(16) glm::vec3 sun_color;
    };

    struct SunUBO {
      glm::mat4 mvp;
      glm::vec4 color;
    };

    struct TessellationConfigUBO {
      float minTessLevel{1.f};
      float maxTessLevel{128.f};
      float minDistance{1.f};
      float maxDistance{1024.f};
      float frustumCullMargin{.4f};
    };

    struct {
      SkyBoxUBO sky_box;
      WaveUBO wave;
      SunUBO sun;
      OceanParams ocean_params;
      TessellationConfigUBO tess_config;
    } ubos;

    struct UniformBuffers {
      vks::Buffer sky_box;
      vks::Buffer wave;
      vks::Buffer sun;
      vks::Buffer ocean_params;
      vks::Buffer tess_config;
    };
    std::array<UniformBuffers, maxConcurrentFrames> uniform_buffers;

    struct Pipelines {
      VkPipeline sky_box{VK_NULL_HANDLE};
      VkPipeline wave{VK_NULL_HANDLE};
      VkPipeline sun{VK_NULL_HANDLE};
    } pipelines;

    struct {
      VkDescriptorSetLayout sky_box{VK_NULL_HANDLE};
      VkDescriptorSetLayout wave{VK_NULL_HANDLE};
      VkDescriptorSetLayout sun{VK_NULL_HANDLE};
    } descriptor_set_layouts;

    struct {
      VkPipelineLayout sky_box{VK_NULL_HANDLE};
      VkPipelineLayout wave{VK_NULL_HANDLE};
      VkPipelineLayout sun{VK_NULL_HANDLE};
    } pipeline_layouts;

    struct DescriptorSets {
      VkDescriptorSet sky_box{VK_NULL_HANDLE};
      VkDescriptorSet wave{VK_NULL_HANDLE};
      VkDescriptorSet sun{VK_NULL_HANDLE};
    };
    std::array<DescriptorSets, maxConcurrentFrames> descriptor_sets;
  } graphics_;

  struct FFTPushConstants {
    int32_t stage;
    int32_t direction;
    int32_t input_is_ping;
    int32_t output_is_ping;
    int32_t inverse;
    int32_t _padding[3];
  };

  void setupDescriptors() {
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

    // Sun
    setLayoutBindings = {
        // Binding 0 : Vertex/fragment shader ubo
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),
    };
    descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr,
                                    &graphics_.descriptor_set_layouts.sun));

    // Wave
    setLayoutBindings = {
        // Binding 0 : MVP UBO
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
                VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),
        // Binding 1 : Tess. Control Config
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
            /*binding id*/ 1),
        // Binding 2 : Ocean params UBO
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
            /*binding id*/ 2),
        // Binding 3 : Height Map
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
            /*binding id*/ 3),
        // Binding 4 : Normal Map
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 4),
        // Binding 5 : Cube map / skybox
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 5),
    };
    descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr,
                                    &graphics_.descriptor_set_layouts.wave));

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

      // Sun
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool, &graphics_.descriptor_set_layouts.sun, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device, &allocInfo, &graphics_.descriptor_sets[i].sun));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              graphics_.descriptor_sets[i].sun,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
              &graphics_.uniform_buffers[i].sun.descriptor),
      };
      vkUpdateDescriptorSets(device,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Wave
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool, &graphics_.descriptor_set_layouts.wave, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device, &allocInfo, &graphics_.descriptor_sets[i].wave));
      writeDescriptorSets = {
          // MVP ubo
          vks::initializers::writeDescriptorSet(
              graphics_.descriptor_sets[i].wave,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
              &graphics_.uniform_buffers[i].wave.descriptor),

          // tess. control ubo
          vks::initializers::writeDescriptorSet(
              graphics_.descriptor_sets[i].wave,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
              &graphics_.uniform_buffers[i].tess_config.descriptor),

          // tess. eval ocean params
          vks::initializers::writeDescriptorSet(
              graphics_.descriptor_sets[i].wave,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2,
              &graphics_.uniform_buffers[i].ocean_params.descriptor),

          // height map
          vks::initializers::writeDescriptorSet(
              graphics_.descriptor_sets[i].wave,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3,
              &compute_.height_map.descriptor),

          // normal map
          vks::initializers::writeDescriptorSet(
              graphics_.descriptor_sets[i].wave,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4,
              &compute_.normal_map.descriptor),

          // cube map / skybox
          vks::initializers::writeDescriptorSet(
              graphics_.descriptor_sets[i].wave,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5,
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
    // Sun
    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &graphics_.descriptor_set_layouts.sun, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo,
                                           nullptr,
                                           &graphics_.pipeline_layouts.sun));
    // Wave
    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &graphics_.descriptor_set_layouts.wave, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo,
                                           nullptr,
                                           &graphics_.pipeline_layouts.wave));

    // Pipeline
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
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_POLYGON_MODE_EXT};
    VkPipelineDynamicStateCreateInfo dynamicState =
        vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
    std::array<VkPipelineShaderStageCreateInfo, 4> shaderStages{};

    // We render the terrain as a grid of quad patches
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
        vks::initializers::pipelineInputAssemblyStateCreateInfo(
            VK_PRIMITIVE_TOPOLOGY_PATCH_LIST, 0, VK_FALSE);
    VkPipelineTessellationStateCreateInfo tessellationState =
        vks::initializers::pipelineTessellationStateCreateInfo(4);
    // Wave mesh tessellation pipeline
    shaderStages[0] = loadShader(getShadersPath() + "wave/wave.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "wave/wave.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    shaderStages[2] = loadShader(getShadersPath() + "wave/wave.tesc.spv",
                                 VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
    shaderStages[3] = loadShader(getShadersPath() + "wave/wave.tese.spv",
                                 VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);

    VkGraphicsPipelineCreateInfo pipelineCI =
        vks::initializers::pipelineCreateInfo();
    pipelineCI.layout = graphics_.pipeline_layouts.wave;
    pipelineCI.pInputAssemblyState = &inputAssemblyState;
    rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
    pipelineCI.pRasterizationState = &rasterizationState;
    pipelineCI.pColorBlendState = &colorBlendState;
    pipelineCI.pMultisampleState = &multisampleState;
    pipelineCI.pViewportState = &viewportState;
    pipelineCI.pDepthStencilState = &depthStencilState;
    pipelineCI.pDynamicState = &dynamicState;
    pipelineCI.pTessellationState = &tessellationState;
    pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCI.pStages = shaderStages.data();

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

    auto bindingDescription = WaterVertex::getBindingDescription();
    auto attributeDescription = WaterVertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputCI =
        vks::initializers::pipelineVertexInputStateCreateInfo();
    vertexInputCI.vertexBindingDescriptionCount = 1;
    vertexInputCI.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescription.size());
    vertexInputCI.pVertexBindingDescriptions = &bindingDescription;
    vertexInputCI.pVertexAttributeDescriptions = attributeDescription.data();
    pipelineCI.pVertexInputState = &vertexInputCI;

    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1,
                                              &pipelineCI, nullptr,
                                              &graphics_.pipelines.wave));

    // Skybox
    pipelineCI.layout = graphics_.pipeline_layouts.sky_box;
    pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState(
        {vkglTF::VertexComponent::Position, vkglTF::VertexComponent::UV});

    depthStencilState.depthWriteEnable = VK_FALSE;
    depthStencilState.depthTestEnable = VK_TRUE;
    rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;
    rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
    // Revert to triangle list topology
    inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    // Remove unused dynamic states (wireframe)
    dynamicStateEnables = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                           VK_DYNAMIC_STATE_POLYGON_MODE_EXT};
    dynamicState =
        vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
    // Reset tessellation state
    pipelineCI.pTessellationState = nullptr;
    pipelineCI.stageCount = 2;
    pipelineCI.pStages = shaderStages.data();
    shaderStages[0] = loadShader(getShadersPath() + "wave/skybox.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "wave/skybox.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1,
                                              &pipelineCI, nullptr,
                                              &graphics_.pipelines.sky_box));

    // Sun
    pipelineCI.layout = graphics_.pipeline_layouts.sun;
    pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState(
        {vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal,
         vkglTF::VertexComponent::UV});
    depthStencilState.depthWriteEnable = VK_FALSE;
    depthStencilState.depthTestEnable = VK_TRUE;
    rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
    inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    // Remove tessellation state
    pipelineCI.pTessellationState = nullptr;
    // Shader stages
    shaderStages[0] = loadShader(getShadersPath() + "wave/sun.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "wave/sun.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineCI.stageCount = 2;
    pipelineCI.pStages = shaderStages.data();
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1,
                                              &pipelineCI, nullptr,
                                              &graphics_.pipelines.sun));
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

      // Wave
      VK_CHECK_RESULT(vulkanDevice->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.wave, sizeof(graphics_.ubos.wave)));
      VK_CHECK_RESULT(buffer.wave.map());

      // Sun
      VK_CHECK_RESULT(
          vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                     &buffer.sun, sizeof(graphics_.ubos.sun)));
      VK_CHECK_RESULT(buffer.sun.map());

      // Wave Params
      VK_CHECK_RESULT(vulkanDevice->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.ocean_params, sizeof(graphics_.ubos.ocean_params)));
      VK_CHECK_RESULT(buffer.ocean_params.map());

      // Tessellation Config
      VK_CHECK_RESULT(vulkanDevice->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.tess_config, sizeof(graphics_.ubos.tess_config)));
      VK_CHECK_RESULT(buffer.tess_config.map());
    }
  }

  void updateComputeUniformBuffers() {
    const auto now = std::chrono::steady_clock::now();
    if (!sim_time_initialized_) {
      last_sim_time_ = now;
      sim_time_initialized_ = true;
    }
    const std::chrono::duration<float> dt = now - last_sim_time_;
    last_sim_time_ = now;
    if (!ui_features.pause_wave) {
      float scale = 1.0f;
      if (ui_features.time_step > 0) {
        scale = 1.0f / static_cast<float>(ui_features.time_step);
      }
      compute_.ubos.ocean.time_patch_chop_height.x += dt.count() * scale;
    }
    compute_.ubos.ocean.time_patch_chop_height.y = ui_features.patch_scale;
    memcpy(compute_.uniform_buffers[currentBuffer].ocean.mapped,
           &compute_.ubos.ocean, sizeof(OceanParams));
  }

  void updateGraphicsUniformBuffers() {
    // Skybox
    graphics_.ubos.sky_box.perspective = camera.matrices.perspective;
    graphics_.ubos.sky_box.view = glm::mat4(glm::mat3(camera.matrices.view));
    memcpy(graphics_.uniform_buffers[currentBuffer].sky_box.mapped,
           &graphics_.ubos.sky_box, sizeof(Graphics::SkyBoxUBO));

    // Wave
    graphics_.ubos.wave.perspective = camera.matrices.perspective;
    graphics_.ubos.wave.view = camera.matrices.view;
    graphics_.ubos.wave.camera_position = camera.position;
    graphics_.ubos.wave.screen_res = glm::vec2(this->width, this->height);
    graphics_.ubos.wave.patch_scale = ui_features.patch_scale;

    // Sun params for wave
    float sunAngle = compute_.ubos.ocean.time_patch_chop_height.x * 0.1f;
    const float sunRadius = 600.0f;
    glm::vec3 basePos = glm::vec3(0.0f, 0.0f, sunRadius);
    glm::mat4 rotX =
        glm::rotate(glm::mat4(1.0f), sunAngle, glm::vec3(1.0f, 0.0f, 0.0f));
    glm::vec3 sunPos = glm::vec3(rotX * glm::vec4(basePos, 1.0f));
    glm::vec3 sunColor = glm::vec3(1.0f, 0.95f, 0.7f);
    graphics_.ubos.wave.sun_position = sunPos;
    graphics_.ubos.wave.sun_color = sunColor;
    memcpy(graphics_.uniform_buffers[currentBuffer].wave.mapped,
           &graphics_.ubos.wave, sizeof(Graphics::WaveUBO));

    // Sun params
    graphics_.ubos.sun.color = glm::vec4(sunColor, 1.0f);
    glm::mat4 sunModel =
        glm::translate(glm::mat4(1.0f), sunPos) *
        glm::scale(glm::mat4(1.0f),
                   glm::vec3(ui_features.sun_scale, ui_features.sun_scale,
                             ui_features.sun_scale));
    graphics_.ubos.sun.mvp =
        graphics_.ubos.wave.perspective * graphics_.ubos.wave.view * sunModel;
    memcpy(graphics_.uniform_buffers[currentBuffer].sun.mapped,
           &graphics_.ubos.sun, sizeof(Graphics::SunUBO));

    // Ocean params
    graphics_.ubos.ocean_params = compute_.ubos.ocean;
    memcpy(graphics_.uniform_buffers[currentBuffer].ocean_params.mapped,
           &compute_.ubos.ocean, sizeof(OceanParams));

    // Tess Config
    memcpy(graphics_.uniform_buffers[currentBuffer].tess_config.mapped,
           &graphics_.ubos.tess_config,
           sizeof(Graphics::TessellationConfigUBO));
  }

  void setupWaterMesh() {
    // 1. Generate the mesh
    WaterMesh waterMesh;
    waterMesh.generateUnitQuad();

    // 2. Create Vulkan buffers
    graphics_.wave_mesh_buffers.index_count = waterMesh.indices.size();
    uint32_t vertex_buffer_size =
        waterMesh.vertices.size() * sizeof(WaterVertex);
    uint32_t index_buffer_size = waterMesh.indices.size() * sizeof(uint32_t);
    vks::Buffer vertexStaging, indexStaging;

    // 3. Store for rendering
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &vertexStaging, vertex_buffer_size, waterMesh.vertices.data()));

    VK_CHECK_RESULT(vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &indexStaging, index_buffer_size, waterMesh.indices.data()));

    VK_CHECK_RESULT(vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &graphics_.wave_mesh_buffers.vertex_buffer, vertex_buffer_size));

    VK_CHECK_RESULT(vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &graphics_.wave_mesh_buffers.index_buffer, index_buffer_size));

    // Copy from staging buffers
    VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    VkBufferCopy copyRegion = {};

    copyRegion.size = vertex_buffer_size;
    vkCmdCopyBuffer(copyCmd, vertexStaging.buffer,
                    graphics_.wave_mesh_buffers.vertex_buffer.buffer, 1,
                    &copyRegion);

    copyRegion.size = index_buffer_size;
    vkCmdCopyBuffer(copyCmd, indexStaging.buffer,
                    graphics_.wave_mesh_buffers.index_buffer.buffer, 1,
                    &copyRegion);

    vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

    vkDestroyBuffer(device, vertexStaging.buffer, nullptr);
    vkFreeMemory(device, vertexStaging.memory, nullptr);
    vkDestroyBuffer(device, indexStaging.buffer, nullptr);
    vkFreeMemory(device, indexStaging.memory, nullptr);
  }

  void prepareDescriptorPool() {
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            /*total ubo count */ (/*graphics*/ 5 + /*compute*/ 1) *
                maxConcurrentFrames),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            /*total texture count (across all pipelines) */ (
                /*graphics*/ 4 + /*compute textures*/ 0) *
                maxConcurrentFrames),
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                              5 * maxConcurrentFrames)};

    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(
            poolSizes,
            /*total descriptor count*/ (/*graphics*/ 3 + /*compute*/ 1) *
                maxConcurrentFrames);
    // Needed if using VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT in
    // descriptor bindings
    descriptorPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr,
                                           &descriptorPool));
  };

  void prepareGraphics() {
    vkCmdSetPolygonModeEXT = reinterpret_cast<PFN_vkCmdSetPolygonModeEXT>(
        vkGetDeviceProcAddr(device, "vkCmdSetPolygonModeEXT"));
    loadAssets();
    setupWaterMesh();
    prepareUniformBuffers();
    setupDescriptors();
    preparePipelines();
  }

  void prepareComputeTextures() {
    // Create a compute capable device queue
    vkGetDeviceQueue(device, compute_.queueFamilyIndex, 0, &compute_.queue);
    compute_.spectrum_initialized = false;
    createComputeTexture(compute_.spectrum_h0, VK_FORMAT_R32G32B32A32_SFLOAT);
    createComputeTexture(compute_.spectrum_ht_ping,
                         VK_FORMAT_R32G32B32A32_SFLOAT);
    createComputeTexture(compute_.spectrum_ht_pong,
                         VK_FORMAT_R32G32B32A32_SFLOAT);
    createComputeTexture(compute_.height_map, VK_FORMAT_R32_SFLOAT);
    createComputeTexture(compute_.normal_map, VK_FORMAT_R16G16B16A16_SFLOAT);
    clearAllComputeTextures();
  }

  void createComputeTexture(Compute::Texture2D& texture,
                            VkFormat texture_format) const {
    // A 2D texture is described as width x height x depth
    texture.width = COMPUTE_TEXTURE_DIMENSION;
    texture.height = COMPUTE_TEXTURE_DIMENSION;
    texture.depth = 1;
    texture.mipLevels = 1;
    texture.format = texture_format;

    // Format support check
    // 2D texture support in Vulkan is mandatory (in contrast to OpenGL) so no
    // need to check if it's supported
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, texture.format,
                                        &formatProperties);
    // Check if format supports transfer
    if (!(formatProperties.optimalTilingFeatures &
          VK_FORMAT_FEATURE_TRANSFER_DST_BIT)) {
      std::cout << "Error: Device does not support flag TRANSFER_DST for "
                   "selected texture format!"
                << std::endl;
      return;
    }

    // Create optimal tiled target image
    VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = texture.format;
    imageCreateInfo.mipLevels = texture.mipLevels;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.extent.width = texture.width;
    imageCreateInfo.extent.height = texture.height;
    imageCreateInfo.extent.depth = 1;
    // Set initial layout of the image to undefined
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.usage =
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

    VK_CHECK_RESULT(
        vkCreateImage(device, &imageCreateInfo, nullptr, &texture.image));

    // Device local memory to back up image
    VkMemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs = {};
    vkGetImageMemoryRequirements(device, texture.image, &memReqs);
    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr,
                                     &texture.deviceMemory));
    VK_CHECK_RESULT(
        vkBindImageMemory(device, texture.image, texture.deviceMemory, 0));

    // Transition read textures
    VkCommandBuffer layoutCmd = vulkanDevice->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    texture.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    vks::tools::setImageLayout(layoutCmd, texture.image,
                               VK_IMAGE_ASPECT_COLOR_BIT,
                               VK_IMAGE_LAYOUT_UNDEFINED, texture.imageLayout);
    if (vulkanDevice->queueFamilyIndices.graphics !=
        vulkanDevice->queueFamilyIndices.compute) {
      VkImageMemoryBarrier imageMemoryBarrier = {};
      imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
      imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      imageMemoryBarrier.image = texture.image;
      imageMemoryBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                             1};
      imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      imageMemoryBarrier.dstAccessMask = 0;
      imageMemoryBarrier.srcQueueFamilyIndex =
          vulkanDevice->queueFamilyIndices.graphics;
      imageMemoryBarrier.dstQueueFamilyIndex =
          vulkanDevice->queueFamilyIndices.compute;
      vkCmdPipelineBarrier(layoutCmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_FLAGS_NONE,
                           0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
    }

    vulkanDevice->flushCommandBuffer(layoutCmd, queue, true);

    VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
    const bool supportsLinearFilter =
        (formatProperties.optimalTilingFeatures &
         VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
    sampler.magFilter =
        supportsLinearFilter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    sampler.minFilter =
        supportsLinearFilter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.mipLodBias = 0.0f;
    sampler.compareOp = VK_COMPARE_OP_NEVER;
    sampler.minLod = 0.0f;
    sampler.maxLod = 0.0f;
    sampler.maxAnisotropy = 1.0;
    sampler.anisotropyEnable = VK_FALSE;
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    VK_CHECK_RESULT(
        vkCreateSampler(device, &sampler, nullptr, &texture.sampler));

    // Create image view
    VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
    view.image = texture.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = texture.format;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.baseMipLevel = 0;
    view.subresourceRange.baseArrayLayer = 0;
    view.subresourceRange.layerCount = 1;
    view.subresourceRange.levelCount = 1;
    VK_CHECK_RESULT(vkCreateImageView(device, &view, nullptr, &texture.view));

    texture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    texture.descriptor.imageView = texture.view;
    texture.descriptor.sampler = texture.sampler;
  }

  void clearAllComputeTextures() const {
    // Clear all textures
    const VkCommandBuffer clearCmd = vulkanDevice->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    constexpr VkClearColorValue clearColor = {{0.0f, 0.0f, 0.0f, 0.0f}};
    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = 0;
    range.levelCount = 1;
    range.baseArrayLayer = 0;
    range.layerCount = 1;
    vkCmdClearColorImage(clearCmd, compute_.spectrum_h0.image,
                         VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);
    vkCmdClearColorImage(clearCmd, compute_.spectrum_ht_ping.image,
                         VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);
    vkCmdClearColorImage(clearCmd, compute_.spectrum_ht_pong.image,
                         VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);
    vkCmdClearColorImage(clearCmd, compute_.height_map.image,
                         VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);
    vkCmdClearColorImage(clearCmd, compute_.normal_map.image,
                         VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);
    vulkanDevice->flushCommandBuffer(clearCmd, queue, true);
  }

  void insertComputeBarrier(const VkCommandBuffer& cmdBuffer) const {
    VkImageMemoryBarrier barriers[5]{};
    auto fillBarrier = [](VkImageMemoryBarrier& barrier, VkImage image) {
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = image;
      barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    };
    fillBarrier(barriers[0], compute_.spectrum_h0.image);
    fillBarrier(barriers[1], compute_.spectrum_ht_ping.image);
    fillBarrier(barriers[2], compute_.spectrum_ht_pong.image);
    fillBarrier(barriers[3], compute_.height_map.image);
    fillBarrier(barriers[4], compute_.normal_map.image);

    vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 5, barriers);
  }

  void prepareComputeUniformBuffers() {
    for (auto& buffer : compute_.uniform_buffers) {
      VK_CHECK_RESULT(vulkanDevice->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.ocean, sizeof(compute_.ubos.ocean)));
      VK_CHECK_RESULT(buffer.ocean.map());
    }
  }

  void prepareComputeDescriptors() {
    // Layouts
    VkDescriptorSetLayoutCreateInfo descriptorLayout;
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;

    // Ocean compute set (shared by all compute pipelines)
    setLayoutBindings = {
        // Binding 0 : h0 spectrum (rw)
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 0),
        // Binding 1 : ht spectrum ping (rw)
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 1),
        // Binding 2 : ht spectrum pong (rw)
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 2),
        // Binding 3 : height map (rw)
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 3),
        // Binding 4 : normal map (rw)
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 4),
        // Binding 5 : UBO
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 5),
    };
    descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr,
                                    &compute_.descriptor_set_layouts.ocean));

    for (auto i = 0; i < compute_.uniform_buffers.size(); i++) {
      // Ocean compute set
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool, &compute_.descriptor_set_layouts.ocean, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device, &allocInfo, &compute_.descriptor_sets[i].ocean));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              compute_.descriptor_sets[i].ocean,
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0,
              &compute_.spectrum_h0.descriptor),
          vks::initializers::writeDescriptorSet(
              compute_.descriptor_sets[i].ocean,
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
              &compute_.spectrum_ht_ping.descriptor),
          vks::initializers::writeDescriptorSet(
              compute_.descriptor_sets[i].ocean,
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2,
              &compute_.spectrum_ht_pong.descriptor),
          vks::initializers::writeDescriptorSet(
              compute_.descriptor_sets[i].ocean,
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3,
              &compute_.height_map.descriptor),
          vks::initializers::writeDescriptorSet(
              compute_.descriptor_sets[i].ocean,
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4,
              &compute_.normal_map.descriptor),
          vks::initializers::writeDescriptorSet(
              compute_.descriptor_sets[i].ocean,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 5,
              &compute_.uniform_buffers[i].ocean.descriptor),
      };
      vkUpdateDescriptorSets(device,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  void prepareComputePipelines() {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(int32_t) * 8;

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &compute_.descriptor_set_layouts.ocean, 1);
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo,
                                           nullptr,
                                           &compute_.pipeline_layouts.ocean));

    VkComputePipelineCreateInfo computePipelineCreateInfo =
        vks::initializers::computePipelineCreateInfo(
            compute_.pipeline_layouts.ocean, 0);

    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "wave/init_spectrum.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, pipelineCache, 1, &computePipelineCreateInfo, nullptr,
        &compute_.pipelines.init_spectrum));

    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "wave/spectrum.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, pipelineCache, 1, &computePipelineCreateInfo, nullptr,
        &compute_.pipelines.spectrum));

    computePipelineCreateInfo.stage = loadShader(
        getShadersPath() + "wave/fft.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1,
                                             &computePipelineCreateInfo,
                                             nullptr, &compute_.pipelines.fft));

    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "wave/resolve_height.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, pipelineCache, 1, &computePipelineCreateInfo, nullptr,
        &compute_.pipelines.resolve_height));

    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "wave/normals.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, pipelineCache, 1, &computePipelineCreateInfo, nullptr,
        &compute_.pipelines.normals));
  }

  void prepareDebugExt() {
    vkCmdBeginDebugUtilsLabelEXT =
        reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
            vkGetInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT"));
    vkCmdEndDebugUtilsLabelEXT =
        reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
            vkGetInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT"));
  }

  void cmdBeginLabel(const VkCommandBuffer& command_buffer,
                     const char* label_name,
                     const std::array<float, 4> color = debugColor_) const {
    if (!vkCmdBeginDebugUtilsLabelEXT) {
      return;
    }
    VkDebugUtilsLabelEXT label = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = label_name;
    memcpy(label.color, color.data(), sizeof(float) * 4);
    vkCmdBeginDebugUtilsLabelEXT(command_buffer, &label);
  }

  void cmdEndLabel(const VkCommandBuffer& command_buffer) const {
    if (!vkCmdEndDebugUtilsLabelEXT) {
      return;
    }
    vkCmdEndDebugUtilsLabelEXT(command_buffer);
  }

  void prepareCompute() {
    prepareComputeTextures();
    prepareComputeUniformBuffers();
    prepareComputeDescriptors();
    prepareComputePipelines();
    prepareComputeCommandPoolBuffersFencesAndSemaphores();
    prepareInitialWaveState();
  }

  void prepareInitialWaveState() {
    const int32_t logN =
        static_cast<int32_t>(std::log2(COMPUTE_TEXTURE_DIMENSION));
    compute_.ubos.ocean.wind_dir_speed_amp =
        glm::vec4(1.0f, 0.0f, 20.0f, 0.05f);
    compute_.ubos.ocean.time_patch_chop_height =
        glm::vec4(0.0f, ui_features.patch_scale, 1.0f, 128.0f);
    compute_.ubos.ocean.grid =
        glm::ivec4(COMPUTE_TEXTURE_DIMENSION, logN, 0, 0);
    for (auto& buffer : compute_.uniform_buffers) {
      memcpy(buffer.ocean.mapped, &compute_.ubos.ocean, sizeof(OceanParams));
    }
  }

  void prepareComputeCommandPoolBuffersFencesAndSemaphores() {
    // Separate command pool as queue family for compute may be different than
    // graphics
    VkCommandPoolCreateInfo cmdPoolInfo =
        vks::initializers::commandPoolCreateInfo();
    cmdPoolInfo.queueFamilyIndex = compute_.queueFamilyIndex;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr,
                                        &compute_.commandPool));

    // Create command buffers for compute operations
    for (auto& cmdBuffer : compute_.commandBuffers) {
      cmdBuffer = vulkanDevice->createCommandBuffer(
          VK_COMMAND_BUFFER_LEVEL_PRIMARY, compute_.commandPool);
    }

    // Fences to check for command buffer completion
    for (auto& fence : compute_.fences) {
      VkFenceCreateInfo fenceCreateInfo =
          vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
      VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));
    }

    // Semaphores to order compute and graphics submissions
    for (auto& [ready, complete] : compute_.semaphores) {
      VkSemaphoreCreateInfo semaphoreInfo{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      vkCreateSemaphore(device, &semaphoreInfo, nullptr, &ready);
      vkCreateSemaphore(device, &semaphoreInfo, nullptr, &complete);
    }
    // Signal first used ready semaphore
    VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
    computeSubmitInfo.signalSemaphoreCount = 1;
    computeSubmitInfo.pSignalSemaphores =
        &compute_.semaphores[maxConcurrentFrames - 1].ready;
    VK_CHECK_RESULT(
        vkQueueSubmit(compute_.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE));
  }

  void prepare() override {
    VulkanExampleBase::prepare();
    prepareDebugExt();
    graphics_.queueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
    compute_.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
    prepareDescriptorPool();
    prepareCompute();
    prepareGraphics();
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

  void buildComputeCommandBuffer() {
    const VkCommandBuffer cmdBuffer = compute_.commandBuffers[currentBuffer];
    const VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    // Acquire ownership of compute images from graphics if queues differ
    if (graphics_.queueFamilyIndex != compute_.queueFamilyIndex) {
      VkImageMemoryBarrier barriers[2]{};
      auto fillBarrier = [&](VkImageMemoryBarrier& barrier, VkImage image) {
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = graphics_.queueFamilyIndex;
        barrier.dstQueueFamilyIndex = compute_.queueFamilyIndex;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      };
      fillBarrier(barriers[0], compute_.height_map.image);
      fillBarrier(barriers[1], compute_.normal_map.image);
      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                           0, nullptr, 2, barriers);
    }

    if (!compute_.spectrum_initialized) {
      cmdBeginLabel(cmdBuffer, "Compute Init Spectrum");
      initSpectrumCmd(cmdBuffer);
      cmdEndLabel(cmdBuffer);
      compute_.spectrum_initialized = true;
      insertComputeBarrier(cmdBuffer);
    }

    cmdBeginLabel(cmdBuffer, "Compute Spectrum");
    spectrumCmd(cmdBuffer);
    cmdEndLabel(cmdBuffer);
    insertComputeBarrier(cmdBuffer);

    cmdBeginLabel(cmdBuffer, "Compute FFT");
    fftCmd(cmdBuffer);
    cmdEndLabel(cmdBuffer);
    insertComputeBarrier(cmdBuffer);

    cmdBeginLabel(cmdBuffer, "Compute Resolve Height");
    resolveHeightCmd(cmdBuffer);
    cmdEndLabel(cmdBuffer);
    insertComputeBarrier(cmdBuffer);

    cmdBeginLabel(cmdBuffer, "Compute Normals");
    normalsCmd(cmdBuffer);
    cmdEndLabel(cmdBuffer);

    // Release ownership of compute images to graphics if queues differ
    if (graphics_.queueFamilyIndex != compute_.queueFamilyIndex) {
      VkImageMemoryBarrier barriers[2]{};
      auto fillBarrier = [&](VkImageMemoryBarrier& barrier, VkImage image) {
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = 0;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = compute_.queueFamilyIndex;
        barrier.dstQueueFamilyIndex = graphics_.queueFamilyIndex;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      };
      fillBarrier(barriers[0], compute_.height_map.image);
      fillBarrier(barriers[1], compute_.normal_map.image);
      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr,
                           0, nullptr, 2, barriers);
    }

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void initSpectrumCmd(const VkCommandBuffer& cmdBuffer) {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines.init_spectrum);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_.pipeline_layouts.ocean, 0, 1,
                            &compute_.descriptor_sets[currentBuffer].ocean, 0,
                            nullptr);
    vkCmdDispatch(
        cmdBuffer,
        compute_.spectrum_h0.width / VulkanExample::Compute::WORKGROUP_SIZE,
        compute_.spectrum_h0.height / VulkanExample::Compute::WORKGROUP_SIZE,
        1);
  }

  void spectrumCmd(const VkCommandBuffer& cmdBuffer) {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines.spectrum);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_.pipeline_layouts.ocean, 0, 1,
                            &compute_.descriptor_sets[currentBuffer].ocean, 0,
                            nullptr);
    vkCmdDispatch(cmdBuffer,
                  compute_.spectrum_ht_ping.width /
                      VulkanExample::Compute::WORKGROUP_SIZE,
                  compute_.spectrum_ht_ping.height /
                      VulkanExample::Compute::WORKGROUP_SIZE,
                  1);
  }

  void fftCmd(const VkCommandBuffer& cmdBuffer) {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines.fft);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_.pipeline_layouts.ocean, 0, 1,
                            &compute_.descriptor_sets[currentBuffer].ocean, 0,
                            nullptr);

    const int32_t logN = compute_.ubos.ocean.grid.y;
    FFTPushConstants pc{};
    pc.inverse = 1;

    int32_t input_is_ping = 1;
    int32_t output_is_ping = 0;

    for (int32_t stage = 0; stage < logN; stage++) {
      pc.stage = stage;
      pc.direction = 0;
      pc.input_is_ping = input_is_ping;
      pc.output_is_ping = output_is_ping;
      vkCmdPushConstants(cmdBuffer, compute_.pipeline_layouts.ocean,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cmdBuffer,
                    compute_.spectrum_ht_ping.width /
                        VulkanExample::Compute::WORKGROUP_SIZE,
                    compute_.spectrum_ht_ping.height /
                        VulkanExample::Compute::WORKGROUP_SIZE,
                    1);
      insertComputeBarrier(cmdBuffer);
      std::swap(input_is_ping, output_is_ping);
    }

    for (int32_t stage = 0; stage < logN; stage++) {
      pc.stage = stage;
      pc.direction = 1;
      pc.input_is_ping = input_is_ping;
      pc.output_is_ping = output_is_ping;
      vkCmdPushConstants(cmdBuffer, compute_.pipeline_layouts.ocean,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cmdBuffer,
                    compute_.spectrum_ht_ping.width /
                        VulkanExample::Compute::WORKGROUP_SIZE,
                    compute_.spectrum_ht_ping.height /
                        VulkanExample::Compute::WORKGROUP_SIZE,
                    1);
      insertComputeBarrier(cmdBuffer);
      std::swap(input_is_ping, output_is_ping);
    }
  }

  void resolveHeightCmd(const VkCommandBuffer& cmdBuffer) {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines.resolve_height);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_.pipeline_layouts.ocean, 0, 1,
                            &compute_.descriptor_sets[currentBuffer].ocean, 0,
                            nullptr);
    FFTPushConstants pc{};
    const int32_t totalStages = compute_.ubos.ocean.grid.y * 2;
    pc.input_is_ping = (totalStages % 2 == 0) ? 1 : 0;
    pc.output_is_ping = 0;
    vkCmdPushConstants(cmdBuffer, compute_.pipeline_layouts.ocean,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(
        cmdBuffer,
        compute_.height_map.width / VulkanExample::Compute::WORKGROUP_SIZE,
        compute_.height_map.height / VulkanExample::Compute::WORKGROUP_SIZE, 1);
  }

  void normalsCmd(const VkCommandBuffer& cmdBuffer) {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines.normals);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_.pipeline_layouts.ocean, 0, 1,
                            &compute_.descriptor_sets[currentBuffer].ocean, 0,
                            nullptr);
    vkCmdDispatch(
        cmdBuffer,
        compute_.normal_map.width / VulkanExample::Compute::WORKGROUP_SIZE,
        compute_.normal_map.height / VulkanExample::Compute::WORKGROUP_SIZE, 1);
  }

  void buildCommandBuffer() {
    const VkCommandBuffer cmdBuffer = drawCmdBuffers[currentBuffer];

    const VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    // Ensure compute writes are visible to graphics sampling
    VkImageMemoryBarrier computeReadBarriers[2]{};
    auto fillComputeReadBarrier = [&](VkImageMemoryBarrier& barrier,
                                      VkImage image) {
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      barrier.srcQueueFamilyIndex = compute_.queueFamilyIndex;
      barrier.dstQueueFamilyIndex = graphics_.queueFamilyIndex;
      barrier.image = image;
      barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    };
    fillComputeReadBarrier(computeReadBarriers[0], compute_.height_map.image);
    fillComputeReadBarrier(computeReadBarriers[1], compute_.normal_map.image);
    if (graphics_.queueFamilyIndex != compute_.queueFamilyIndex) {
      vkCmdPipelineBarrier(
          cmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
          VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT |
              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
          0, 0, nullptr, 0, nullptr, 2, computeReadBarriers);
    } else {
      // Same queue family: keep ownership ignored and use proper access sync
      for (auto& barrier : computeReadBarriers) {
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      }
      vkCmdPipelineBarrier(
          cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT |
              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
          0, 0, nullptr, 0, nullptr, 2, computeReadBarriers);
    }

    // With dynamic rendering there are no subpass dependencies, so we need to
    // take care of proper layout transitions by using barriers. This set of
    // barriers prepares the color and depth images for output.
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, swapChain.images[currentImageIndex],
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, depthStencil.image,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
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
    cmdBeginLabel(cmdBuffer, "Graphics Skybox");
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            graphics_.pipeline_layouts.sky_box, 0, 1,
                            &graphics_.descriptor_sets[currentBuffer].sky_box,
                            0, nullptr);
    vkCmdSetPolygonModeEXT(cmdBuffer, VK_POLYGON_MODE_FILL);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics_.pipelines.sky_box);
    graphics_.models.sky_box.draw(cmdBuffer);
    cmdEndLabel(cmdBuffer);

    // Sun
    cmdBeginLabel(cmdBuffer, "Graphics Sun");
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            graphics_.pipeline_layouts.sun, 0, 1,
                            &graphics_.descriptor_sets[currentBuffer].sun, 0,
                            nullptr);
    vkCmdSetPolygonModeEXT(cmdBuffer, VK_POLYGON_MODE_FILL);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics_.pipelines.sun);
    graphics_.models.sun.draw(cmdBuffer);
    cmdEndLabel(cmdBuffer);

    // Wave
    cmdBeginLabel(cmdBuffer, "Graphics Wave");
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            graphics_.pipeline_layouts.wave, 0, 1,
                            &graphics_.descriptor_sets[currentBuffer].wave, 0,
                            nullptr);
    if (ui_features.show_mesh_only) {
      vkCmdSetPolygonModeEXT(cmdBuffer, VK_POLYGON_MODE_LINE);
    } else {
      vkCmdSetPolygonModeEXT(cmdBuffer, VK_POLYGON_MODE_FILL);
    }
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics_.pipelines.wave);
    VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1,
                           &graphics_.wave_mesh_buffers.vertex_buffer.buffer,
                           offsets);
    vkCmdDraw(cmdBuffer, 4, 1, 0, 0);
    cmdEndLabel(cmdBuffer);

    drawUI(cmdBuffer);

    // End dynamic rendering
    vkCmdEndRendering(cmdBuffer);

    // Transition to present
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, swapChain.images[currentImageIndex],
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    // Release ownership back to compute if queues differ
    if (graphics_.queueFamilyIndex != compute_.queueFamilyIndex) {
      VkImageMemoryBarrier barriers[2]{};
      auto fillBarrier = [&](VkImageMemoryBarrier& barrier, VkImage image) {
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = 0;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = graphics_.queueFamilyIndex;
        barrier.dstQueueFamilyIndex = compute_.queueFamilyIndex;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      };
      fillBarrier(barriers[0], compute_.height_map.image);
      fillBarrier(barriers[1], compute_.normal_map.image);
      vkCmdPipelineBarrier(
          cmdBuffer,
          VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT |
              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
          VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 2,
          barriers);
    }

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void submitFrameWithComputeWait() {
    const VkSemaphore waitSemaphores[] = {
        presentCompleteSemaphores[currentBuffer],
        compute_.semaphores[currentBuffer].complete};
    const VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT};

    VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = static_cast<uint32_t>(std::size(waitSemaphores)),
        .pWaitSemaphores = waitSemaphores,
        .pWaitDstStageMask = waitStages,
        .commandBufferCount = 1,
        .pCommandBuffers = &drawCmdBuffers[currentBuffer],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &renderCompleteSemaphores[currentImageIndex]};

    VK_CHECK_RESULT(vkResetFences(device, 1, &waitFences[currentBuffer]));
    VK_CHECK_RESULT(
        vkQueueSubmit(queue, 1, &submitInfo, waitFences[currentBuffer]));

    VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &renderCompleteSemaphores[currentImageIndex],
        .swapchainCount = 1,
        .pSwapchains = &swapChain.swapChain,
        .pImageIndices = &currentImageIndex};
    VkResult result = vkQueuePresentKHR(queue, &presentInfo);
    if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR)) {
      windowResize();
      if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        return;
      }
    } else {
      VK_CHECK_RESULT(result);
    }
    currentBuffer = (currentBuffer + 1) % maxConcurrentFrames;
  }

  void render() override {
    if (!prepared) {
      return;
    }
    const bool sameQueueFamily =
        (graphics_.queueFamilyIndex == compute_.queueFamilyIndex);

    if (!sameQueueFamily) {
      // Ensure previous graphics work that reads compute outputs is finished
      VK_CHECK_RESULT(vkWaitForFences(device, 1, &waitFences[currentBuffer],
                                      VK_TRUE, UINT64_MAX));
      VK_CHECK_RESULT(vkResetFences(device, 1, &waitFences[currentBuffer]));
    }

    // Compute
    // Use a fence to ensure that compute command buffer has finished
    // executing before using it again
    VK_CHECK_RESULT(vkWaitForFences(device, 1, &compute_.fences[currentBuffer],
                                    VK_TRUE, UINT64_MAX));
    VK_CHECK_RESULT(vkResetFences(device, 1, &compute_.fences[currentBuffer]));
    updateComputeUniformBuffers();
    buildComputeCommandBuffer();

    VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
    computeSubmitInfo.commandBufferCount = 1;
    computeSubmitInfo.pCommandBuffers = &compute_.commandBuffers[currentBuffer];
    if (!sameQueueFamily) {
      computeSubmitInfo.signalSemaphoreCount = 1;
      computeSubmitInfo.pSignalSemaphores =
          &compute_.semaphores[currentBuffer].complete;
    }
    VK_CHECK_RESULT(vkQueueSubmit(compute_.queue, 1, &computeSubmitInfo,
                                  compute_.fences[currentBuffer]));

    // Graphics
    if (sameQueueFamily) {
      VulkanExampleBase::prepareFrame();
      updateGraphicsUniformBuffers();
      buildCommandBuffer();
      VulkanExampleBase::submitFrame();
    } else {
      VulkanExampleBase::prepareFrame(false);
      updateGraphicsUniformBuffers();
      buildCommandBuffer();
      submitFrameWithComputeWait();
    }
  }

  void OnUpdateUIOverlay(vks::UIOverlay* overlay) override {
    if (deviceFeatures.fillModeNonSolid) {
      if (overlay->checkBox("VSync", &settings.vsync)) {
        windowResize();
      }
      overlay->checkBox("Pause", &ui_features.pause_wave);
      overlay->checkBox("Show Mesh Only", &ui_features.show_mesh_only);
      overlay->sliderFloat("Time Step", &ui_features.time_step, 0.01, 2);
      overlay->sliderFloat("Patch Scale", &ui_features.patch_scale, 10,
                           1 << (10));
      overlay->sliderFloat("Sun Scale", &ui_features.sun_scale, 10.0f, 200.0f);
      overlay->sliderFloat("Tess Min Level",
                           &graphics_.ubos.tess_config.minTessLevel, 1.0f,
                           64.0f);
      overlay->sliderFloat("Tess Max Level",
                           &graphics_.ubos.tess_config.maxTessLevel, 1.0f,
                           512.0f);
      overlay->sliderFloat("Tess Min Distance",
                           &graphics_.ubos.tess_config.minDistance, 1.0f,
                           1024.0f);
      overlay->sliderFloat("Tess Max Distance",
                           &graphics_.ubos.tess_config.maxDistance, 1.0f,
                           4096.0f);
      overlay->sliderFloat(
          "Chop", &compute_.ubos.ocean.time_patch_chop_height.z, 0.0f, 10.0f);
      overlay->sliderFloat("Height Scale",
                           &compute_.ubos.ocean.time_patch_chop_height.w, 0.0f,
                           512.0f);
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
    graphics_.models.sun.loadFromFile(getAssetPath() + "models/sphere.gltf",
                                      vulkanDevice, queue, glTFLoadingFlags);
    // Skybox textures
    graphics_.textures.cube_map.loadFromFile(
        getExamplesBasePath() + "wave/cartoon_skybox.ktx",
        VK_FORMAT_R8G8B8A8_SRGB, vulkanDevice, queue);
  }

  // Enable physical device features required for this example
  void getEnabledFeatures() override {
    // Tessellation shader support is required for this example
    if (deviceFeatures.tessellationShader) {
      enabledFeatures.tessellationShader = VK_TRUE;
    } else {
      vks::tools::exitFatal(
          "Selected GPU does not support tessellation shaders!",
          VK_ERROR_FEATURE_NOT_PRESENT);
    }
    // Fill mode non-solid is required for wireframe display
    if (deviceFeatures.fillModeNonSolid) {
      enabledFeatures.fillModeNonSolid = VK_TRUE;
    };
  }

  VulkanExample() : VulkanExampleBase() {
    title = "Wave Simulation";
    camera.type_ = Camera::CameraType::firstperson;
    camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 1024.0f);
    camera.setTranslation(glm::vec3(0.0f, 20.0f, 0.0f));
    camera.movementSpeed = 100.0f;

    apiVersion = VK_API_VERSION_1_3;

    // Dynamic rendering
    enabledFeatures13_.dynamicRendering = VK_TRUE;
    enabledFeatures13_.pNext = &dynamicState3Features;

    // Dynamic states
    dynamicState3Features.extendedDynamicState3PolygonMode = VK_TRUE;
    enabledDeviceExtensions.push_back(
        VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);

    deviceCreatepNextChain = &enabledFeatures13_;
  }

  ~VulkanExample() override {
    if (device) {
      // Pipelines
      vkDestroyPipeline(device, graphics_.pipelines.sky_box, nullptr);
      vkDestroyPipeline(device, graphics_.pipelines.wave, nullptr);
      vkDestroyPipeline(device, graphics_.pipelines.sun, nullptr);
      vkDestroyPipeline(device, compute_.pipelines.init_spectrum, nullptr);
      vkDestroyPipeline(device, compute_.pipelines.spectrum, nullptr);
      vkDestroyPipeline(device, compute_.pipelines.fft, nullptr);
      vkDestroyPipeline(device, compute_.pipelines.resolve_height, nullptr);
      vkDestroyPipeline(device, compute_.pipelines.normals, nullptr);

      // Pipeline Layouts
      vkDestroyPipelineLayout(device, graphics_.pipeline_layouts.sky_box,
                              nullptr);
      vkDestroyPipelineLayout(device, graphics_.pipeline_layouts.sun, nullptr);
      vkDestroyPipelineLayout(device, graphics_.pipeline_layouts.wave, nullptr);
      vkDestroyPipelineLayout(device, compute_.pipeline_layouts.ocean, nullptr);

      // Buffers: Graphics
      for (auto& buffer : graphics_.uniform_buffers) {
        buffer.sky_box.destroy();
        buffer.wave.destroy();
        buffer.sun.destroy();
        buffer.ocean_params.destroy();
        buffer.tess_config.destroy();
      }
      // Buffers: Compose
      for (auto& buffer : compute_.uniform_buffers) {
        buffer.ocean.destroy();
      }

      // Vertex buffers
      graphics_.wave_mesh_buffers.vertex_buffer.destroy();
      graphics_.wave_mesh_buffers.index_buffer.destroy();

      // Descriptor Layouts
      vkDestroyDescriptorSetLayout(
          device, graphics_.descriptor_set_layouts.sky_box, nullptr);
      vkDestroyDescriptorSetLayout(device, graphics_.descriptor_set_layouts.sun,
                                   nullptr);
      vkDestroyDescriptorSetLayout(
          device, graphics_.descriptor_set_layouts.wave, nullptr);
      vkDestroyDescriptorSetLayout(
          device, compute_.descriptor_set_layouts.ocean, nullptr);

      // Cube map
      graphics_.textures.cube_map.destroy();

      // Compute textures
      vkDestroyImageView(device, compute_.spectrum_h0.view, nullptr);
      vkDestroyImage(device, compute_.spectrum_h0.image, nullptr);
      vkDestroySampler(device, compute_.spectrum_h0.sampler, nullptr);
      vkFreeMemory(device, compute_.spectrum_h0.deviceMemory, nullptr);

      vkDestroyImageView(device, compute_.spectrum_ht_ping.view, nullptr);
      vkDestroyImage(device, compute_.spectrum_ht_ping.image, nullptr);
      vkDestroySampler(device, compute_.spectrum_ht_ping.sampler, nullptr);
      vkFreeMemory(device, compute_.spectrum_ht_ping.deviceMemory, nullptr);

      vkDestroyImageView(device, compute_.spectrum_ht_pong.view, nullptr);
      vkDestroyImage(device, compute_.spectrum_ht_pong.image, nullptr);
      vkDestroySampler(device, compute_.spectrum_ht_pong.sampler, nullptr);
      vkFreeMemory(device, compute_.spectrum_ht_pong.deviceMemory, nullptr);

      vkDestroyImageView(device, compute_.height_map.view, nullptr);
      vkDestroyImage(device, compute_.height_map.image, nullptr);
      vkDestroySampler(device, compute_.height_map.sampler, nullptr);
      vkFreeMemory(device, compute_.height_map.deviceMemory, nullptr);

      vkDestroyImageView(device, compute_.normal_map.view, nullptr);
      vkDestroyImage(device, compute_.normal_map.image, nullptr);
      vkDestroySampler(device, compute_.normal_map.sampler, nullptr);
      vkFreeMemory(device, compute_.normal_map.deviceMemory, nullptr);

      // Compute Commands
      vkDestroyCommandPool(device, compute_.commandPool, nullptr);
      for (const auto& fence : compute_.fences) {
        vkDestroyFence(device, fence, nullptr);
      }
      for (auto& [ready, complete] : compute_.semaphores) {
        vkDestroySemaphore(device, ready, nullptr);
        vkDestroySemaphore(device, complete, nullptr);
      }
    }
  }
};

VULKAN_EXAMPLE_MAIN()
