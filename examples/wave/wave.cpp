/*
 * Vulkan Example - Wave Simulation  *
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#include <ktxvulkan.h>

#include <array>
#include <cmath>
#include <random>
#include <vector>

#include "VulkanglTFModel.h"
#include "frustum.hpp"
#include "vulkanexamplebase.h"

#define VECTOR_FIELD_FORMAT VK_FORMAT_R32G32B32A32_SFLOAT

constexpr uint32_t COMPUTE_TEXTURE_DIMENSION = 256;

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

  // Generate patch list for tessellation (4 vertices per patch)
  void generatePatchGrid(const uint32_t gridSize, const float worldSize) {
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
          vertex.uv[0] = gridX;
          vertex.uv[1] = gridZ;

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

struct WaveParams {
  // Wave parameters organized for efficient GPU access (vec4 alignment)
  glm::vec4 frequency[4];   // 16 waves total (4 frequencies per vec4)
  glm::vec4 amplitude[4];   // Amplitudes for all 16 waves
  glm::vec4 directionX[4];  // X components of wave directions
  glm::vec4 directionY[4];  // Y components of wave directions
  glm::vec4 phase[4];       // Phase offsets for all 16 waves

  // Global parameters
  float time;           // Current simulation time
  float chopiness;      // Wave sharpness (Gerstner chop factor)
  float noiseStrength;  // Noise contribution
  float rippleScale;    // UV scaling for normal map

  // Wind/wave generation parameters
  glm::vec2 windDirection;    // Primary wind direction
  float angleDeviation;       // Max angle deviation from wind (degrees)
  float speedDeviation;       // Random speed variation
                              // Physical constants
  float gravity;              // Gravity constant (affects wave speed)
  float minWavelength;        // Minimum wave length
  float maxWavelength;        // Maximum wave length
  float amplitudeOverLength;  // Ratio of amplitude to wavelength
};

class WaveGenerator {
 private:
  static constexpr int NUM_WAVES = 16;
  static constexpr float PI = 3.14159265359f;
  static constexpr float GRAVITY = 9.81f;  // Or 30.0f from original code

  std::mt19937 rng;
  std::uniform_real_distribution<float> dist;

  void generateWaves(WaveParams& params) {
    const float bumpTexSize = 1.f * COMPUTE_TEXTURE_DIMENSION;

    for (int i = 0; i < NUM_WAVES; ++i) {
      // Calculate which vec4 and component this wave belongs to
      int vec4Index = i / 4;
      int component = i % 4;

      // 1. Calculate wave direction with random deviation
      // Original code: InitTexWave
      float angleRads =
          randomMinusOneToOne() * params.angleDeviation * PI / 180.0f;

      float dx = std::sin(angleRads);
      float dy = std::cos(angleRads);

      // Rotate by wind direction
      float tx = dx;
      dx = params.windDirection.y * dx - params.windDirection.x * dy;
      dy = params.windDirection.x * tx + params.windDirection.y * dy;

      // 2. Calculate wavelength (linearly distributed across range)
      float wavelength = float(i) / float(NUM_WAVES - 1) *
                             (params.maxWavelength - params.minWavelength) +
                         params.minWavelength;

      // Convert to texture space
      float maxLen = params.maxWavelength * bumpTexSize / params.rippleScale;
      float minLen = params.minWavelength * bumpTexSize / params.rippleScale;
      float len = float(i) / float(NUM_WAVES - 1) * (maxLen - minLen) + minLen;

      // 3. Quantize direction to texture pixels
      float reps = bumpTexSize / len;
      dx *= reps;
      dy *= reps;
      dx = std::round(dx);  // Snap to integers
      dy = std::round(dy);

      // Store direction (normalized will happen in shader if needed)
      params.directionX[vec4Index][component] = dx;
      params.directionY[vec4Index][component] = dy;

      // 4. Calculate effective wavelength after quantization
      float effectiveK = 1.0f / std::sqrt(dx * dx + dy * dy);
      float effectiveLen = bumpTexSize * effectiveK;

      // 5. Calculate frequency (k = 2π/λ)
      params.frequency[vec4Index][component] = 2.0f * PI / effectiveLen;

      // 6. Calculate amplitude (proportional to wavelength)
      params.amplitude[vec4Index][component] =
          effectiveLen * params.amplitudeOverLength;

      // 7. Random initial phase
      params.phase[vec4Index][component] = randomZeroToOne();

      // Note: Speed calculation for animation
      // ω = √(gk) where g is gravity, k is frequency
      // For dispersion relation: ω = √(g * 2π/λ)
      // Speed = ω/k = √(g/(2π/λ)) = √(gλ/(2π))
      // This is handled in update function
    }
  }

 public:
  WaveGenerator() : rng(std::random_device{}()), dist(-1.0f, 1.0f) {}

  float randomMinusOneToOne() { return dist(rng); }

  float randomZeroToOne() { return (dist(rng) + 1.0f) * 0.5f; }

  // Initialize wave parameters with physically-based values
  WaveParams initializeWaveParams() {
    WaveParams params{};

    // Global configuration (matching original code's InitTexState)
    params.time = 0.0f;
    params.chopiness = 1.0f;      // Original: m_TexState.m_Chop
    params.noiseStrength = 0.2f;  // Original: m_TexState.m_Noise
    params.rippleScale = 25.0f;   // Original: m_TexState.m_RippleScale

    // Wind parameters
    params.windDirection = glm::vec2(0.0f, 1.0f);  // North
    params.angleDeviation = 15.0f;  // Original: m_TexState.m_AngleDeviation
    params.speedDeviation = 0.1f;   // Original: m_TexState.m_SpeedDeviation

    // Physical parameters
    params.gravity = 30.0f;             // Original: kGravConst
    params.minWavelength = 1.0f;        // Original: m_TexState.m_MinLength
    params.maxWavelength = 10.0f;       // Original: m_TexState.m_MaxLength
    params.amplitudeOverLength = 0.1f;  // Original: m_TexState.m_AmpOverLen

    // Generate individual waves
    generateWaves(params);

    return params;
  }

  // Update wave parameters each frame
  void updateWaveParams(WaveParams& params, float deltaTime) {
    params.time += deltaTime;

    // Update phases based on dispersion relation: ω = √(gk)
    for (int i = 0; i < NUM_WAVES; ++i) {
      int vec4Index = i / 4;
      int component = i % 4;

      float freq = params.frequency[vec4Index][component];
      float wavelength = 2.0f * PI / freq;

      // Dispersion relation for deep water waves: ω = √(gk)
      // But original code uses: speed = √(λ/(2π*g)) / 3
      float speed = std::sqrt(wavelength / (2.0f * PI * params.gravity)) / 3.0f;

      // Apply random speed deviation
      float speedVariation =
          1.0f + randomMinusOneToOne() * params.speedDeviation;
      speed *= speedVariation;

      // Update phase (phase decreases over time in original)
      params.phase[vec4Index][component] -= deltaTime * speed;

      // Keep phase in [0, 1] range
      params.phase[vec4Index][component] =
          std::fmod(params.phase[vec4Index][component] + 1.0f, 1.0f);
    }
  }

  // Calm water
  static WaveParams createCalmWater() {
    WaveGenerator gen;
    WaveParams params = gen.initializeWaveParams();

    params.minWavelength = 2.0f;
    params.maxWavelength = 8.0f;
    params.amplitudeOverLength = 0.05f;  // Small waves
    params.angleDeviation = 10.0f;       // Uniform direction
    params.noiseStrength = 0.1f;
    params.chopiness = 0.5f;

    gen.generateWaves(params);
    return params;
  }

  // Stormy ocean
  static WaveParams createStormyOcean() {
    WaveGenerator gen;
    WaveParams params = gen.initializeWaveParams();

    params.minWavelength = 5.0f;
    params.maxWavelength = 25.0f;
    params.amplitudeOverLength = 0.15f;  // Larger waves
    params.angleDeviation = 30.0f;       // Chaotic directions
    params.noiseStrength = 0.4f;
    params.chopiness = 2.0f;  // Sharp peaks

    gen.generateWaves(params);
    return params;
  }

  // Original DirectX demo settings
  static WaveParams createOriginalSettings() {
    WaveGenerator gen;
    WaveParams params = gen.initializeWaveParams();
    // Texture waves (for bump map)
    params.minWavelength = 1.0f;
    params.maxWavelength = 10.0f;
    params.amplitudeOverLength = 0.1f;
    params.angleDeviation = 15.0f;
    params.windDirection = glm::vec2(0.0f, 1.0f);
    params.noiseStrength = 0.2f;
    params.chopiness = 1.0f;
    params.rippleScale = 25.0f;
    params.speedDeviation = 0.1f;
    params.gravity = 30.0f;

    gen.generateWaves(params);
    return params;
  }
};

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

  // Handles all compute pipelines
  struct Compute {
    static constexpr int WORKGROUP_SIZE = 16;
    static constexpr float TIME_DELTA = 1.f / 360;

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

    Texture2D wave_normal_map;

    struct {
      WaveParams compose;
    } ubos;

    struct UniformBuffers {
      vks::Buffer compose;
    };

    // Buffers
    std::array<UniformBuffers, maxConcurrentFrames> uniform_buffers;

    // Pipelines
    struct {
      VkPipeline compose;
    } pipelines{};

    // Pipeline Layout
    struct {
      VkPipelineLayout compose{VK_NULL_HANDLE};
    } pipeline_layouts;

    // Descriptor Layout
    struct {
      VkDescriptorSetLayout compose{VK_NULL_HANDLE};
    } descriptor_set_layouts;

    // Descriptor Sets
    struct DescriptorSets {
      VkDescriptorSet compose{VK_NULL_HANDLE};
    };
    std::array<DescriptorSets, maxConcurrentFrames> descriptor_sets{};

    WaveGenerator wave_generator;

  } compute_;

  // Handles graphics rendering pipelines
  struct Graphics {
    // families differ and require additional barriers
    uint32_t queueFamilyIndex{0};
    uint32_t GRID_SIZE = 32;
    uint32_t PATCH_COUNT = GRID_SIZE * GRID_SIZE;
    float GRID_SCALE = 100.0f;

    struct {
      vks::Buffer vertex_buffer;
      vks::Buffer index_buffer;
      uint32_t index_count{};
    } wave_mesh_buffers;

    struct {
      vkglTF::Model sky_box{};
    } models;

    struct {
      vks::TextureCubeMap cube_map{};
    } textures;

    struct SkyBoxUBO {
      glm::mat4 perspective;
      glm::mat4 view;
    };

    struct WaveUBO {
      glm::mat4 perspective;
      glm::mat4 view;
      alignas(16) glm::vec3 camera_position;
      glm::vec2 screen_res;
      float pixels_per_edge{20.f};
    };

    struct TessellationConfigUBO {
      float minTessLevel{1.f};
      float maxTessLevel{16.f};
      float minDistance{1.f};
      float maxDistance{200.f};
      float frustumCullMargin{1.f};
    };

    struct {
      SkyBoxUBO sky_box;
      WaveUBO wave;
      WaveParams wave_params;
      TessellationConfigUBO tess_config;
    } ubos;

    struct UniformBuffers {
      vks::Buffer sky_box;
      vks::Buffer wave;
      vks::Buffer wave_params;
      vks::Buffer tess_config;
    };
    std::array<UniformBuffers, maxConcurrentFrames> uniform_buffers;

    struct Pipelines {
      VkPipeline sky_box{VK_NULL_HANDLE};
      VkPipeline wave{VK_NULL_HANDLE};
    } pipelines;

    struct {
      VkDescriptorSetLayout sky_box{VK_NULL_HANDLE};
      VkDescriptorSetLayout wave{VK_NULL_HANDLE};
    } descriptor_set_layouts;

    struct {
      VkPipelineLayout sky_box{VK_NULL_HANDLE};
      VkPipelineLayout wave{VK_NULL_HANDLE};
    } pipeline_layouts;

    struct DescriptorSets {
      VkDescriptorSet sky_box{VK_NULL_HANDLE};
      VkDescriptorSet wave{VK_NULL_HANDLE};
    };
    std::array<DescriptorSets, maxConcurrentFrames> descriptor_sets;
  } graphics_;

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

    // Wave
    setLayoutBindings = {
        // Binding 0 : MVP UBO
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
            /*binding id*/ 0),
        // Binding 1 : Tess. Control Config
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
            /*binding id*/ 1),
        // Binding 2 : WaveParams UBO
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
            /*binding id*/ 2),
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

      // Wave
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool, &graphics_.descriptor_set_layouts.wave, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device, &allocInfo, &graphics_.descriptor_sets[i].wave));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              graphics_.descriptor_sets[i].wave,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
              &graphics_.uniform_buffers[i].wave.descriptor),

          vks::initializers::writeDescriptorSet(
              graphics_.descriptor_sets[i].wave,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
              &graphics_.uniform_buffers[i].tess_config.descriptor),

          vks::initializers::writeDescriptorSet(
              graphics_.descriptor_sets[i].wave,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2,
              &graphics_.uniform_buffers[i].wave_params.descriptor),
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
    // Wave
    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &graphics_.descriptor_set_layouts.wave, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo,
                                           nullptr,
                                           &graphics_.pipeline_layouts.wave));

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

      // Wave Params
      VK_CHECK_RESULT(vulkanDevice->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.wave_params, sizeof(graphics_.ubos.wave_params)));
      VK_CHECK_RESULT(buffer.wave_params.map());

      // Tessellation Config
      VK_CHECK_RESULT(vulkanDevice->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.tess_config, sizeof(graphics_.ubos.tess_config)));
      VK_CHECK_RESULT(buffer.tess_config.map());
    }
  }

  void updateUniformBuffers() {
    // Compute: Compose
    compute_.wave_generator.updateWaveParams(compute_.ubos.compose,
                                             compute_.TIME_DELTA);
    memcpy(compute_.uniform_buffers[currentBuffer].compose.mapped,
           &compute_.ubos.compose, sizeof(WaveParams));

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
    memcpy(graphics_.uniform_buffers[currentBuffer].wave.mapped,
           &graphics_.ubos.wave, sizeof(Graphics::WaveUBO));

    // Wave params
    // NOTE: WaveParams are shared from compute AFTER updating
    memcpy(graphics_.uniform_buffers[currentBuffer].wave_params.mapped,
           &compute_.ubos.compose, sizeof(WaveParams));

    // Tess Config
    memcpy(graphics_.uniform_buffers[currentBuffer].tess_config.mapped,
           &graphics_.ubos.tess_config,
           sizeof(Graphics::TessellationConfigUBO));
  }

  void setupWaterMesh() {
    // 1. Generate the mesh
    WaterMesh waterMesh;
    waterMesh.generatePatchGrid(graphics_.GRID_SIZE, graphics_.GRID_SCALE);

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
            /*total ubo count */ (/*graphics*/ 3 + /*compute*/ 1) *
                maxConcurrentFrames),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            /*total texture count (across all pipelines) */ (
                /*graphics*/ 1 + /*compute textures*/ 0) *
                maxConcurrentFrames),
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                              1 * maxConcurrentFrames)};

    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(
            poolSizes,
            /*total descriptor count*/ (/*graphics*/ 2 + /*compute*/ 1) *
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
    createComputeTexture(compute_.wave_normal_map, VECTOR_FIELD_FORMAT);
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
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_STORAGE_BIT;

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
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
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
    vkCmdClearColorImage(clearCmd, compute_.wave_normal_map.image,
                         VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);
    vulkanDevice->flushCommandBuffer(clearCmd, queue, true);
  }

  void prepareComputeUniformBuffers() {
    for (auto& buffer : compute_.uniform_buffers) {
      VK_CHECK_RESULT(vulkanDevice->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.compose, sizeof(compute_.ubos.compose)));
      VK_CHECK_RESULT(buffer.compose.map());
    }
  }

  void prepareComputeDescriptors() {
    // Layouts
    VkDescriptorSetLayoutCreateInfo descriptorLayout;
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;

    // Compose
    setLayoutBindings = {
        // Binding 0 : Wave normal map (write only)
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 0),
        // Binding 1 : Ubo
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 1),
    };
    descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr,
                                    &compute_.descriptor_set_layouts.compose));

    for (auto i = 0; i < compute_.uniform_buffers.size(); i++) {
      // Normal
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool, &compute_.descriptor_set_layouts.compose, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device, &allocInfo, &compute_.descriptor_sets[i].compose));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              compute_.descriptor_sets[i].compose,
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0,
              &compute_.wave_normal_map.descriptor),
          vks::initializers::writeDescriptorSet(
              compute_.descriptor_sets[i].compose,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
              &compute_.uniform_buffers[i].compose.descriptor),
      };
      vkUpdateDescriptorSets(device,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  void prepareComputePipelines() {
    // Create pipelines
    // Compose
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &compute_.descriptor_set_layouts.compose, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo,
                                           nullptr,
                                           &compute_.pipeline_layouts.compose));

    // Compose
    VkComputePipelineCreateInfo computePipelineCreateInfo =
        vks::initializers::computePipelineCreateInfo(
            compute_.pipeline_layouts.compose, 0);
    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "wave/compose.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);

    // We want to use as much shared memory for the compute shader invocations
    // as available, so we calculate it based on the device limits and pass it
    // to the shader via specialization constants
    uint32_t sharedDataSize = std::min(
        static_cast<uint32_t>(1024),
        static_cast<uint32_t>(
            (vulkanDevice->properties.limits.maxComputeSharedMemorySize /
             sizeof(glm::vec4))));
    VkSpecializationMapEntry specializationMapEntry =
        vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
    VkSpecializationInfo specializationInfo =
        vks::initializers::specializationInfo(1, &specializationMapEntry,
                                              sizeof(int32_t), &sharedDataSize);
    computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;

    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, pipelineCache, 1, &computePipelineCreateInfo, nullptr,
        &compute_.pipelines.compose));
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
    compute_.ubos.compose = compute_.wave_generator.initializeWaveParams();
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

    composeCmd(cmdBuffer);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void composeCmd(const VkCommandBuffer& cmdBuffer) {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines.compose);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_.pipeline_layouts.compose, 0, 1,
                            &compute_.descriptor_sets[currentBuffer].compose, 0,
                            nullptr);
    vkCmdDispatch(
        cmdBuffer,
        compute_.wave_normal_map.width / VulkanExample::Compute::WORKGROUP_SIZE,
        compute_.wave_normal_map.height /
            VulkanExample::Compute::WORKGROUP_SIZE,
        1);
  }

  void buildCommandBuffer() {
    const VkCommandBuffer cmdBuffer = drawCmdBuffers[currentBuffer];

    const VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    // With dynamic rendering there are no subpass dependencies, so we need to
    // take care of proper layout transitions by using barriers This set of
    // barriers prepares the color and depth images for output
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, swapChain.images[currentImageIndex], 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
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
    vkCmdSetPolygonModeEXT(cmdBuffer, VK_POLYGON_MODE_FILL);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics_.pipelines.sky_box);
    graphics_.models.sky_box.draw(cmdBuffer);

    // Wave
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            graphics_.pipeline_layouts.wave, 0, 1,
                            &graphics_.descriptor_sets[currentBuffer].wave, 0,
                            nullptr);
    vkCmdSetPolygonModeEXT(cmdBuffer, VK_POLYGON_MODE_LINE);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics_.pipelines.wave);
    VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1,
                           &graphics_.wave_mesh_buffers.vertex_buffer.buffer,
                           offsets);
    vkCmdBindIndexBuffer(cmdBuffer,
                         graphics_.wave_mesh_buffers.index_buffer.buffer, 0,
                         VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmdBuffer, graphics_.wave_mesh_buffers.index_count, 1, 0,
                     0, 0);

    drawUI(cmdBuffer);

    // End dynamic rendering
    vkCmdEndRendering(cmdBuffer);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void render() override {
    if (!prepared) {
      return;
    }
    {
      // Compute
      // Use a fence to ensure that compute command buffer has finished
      // executing before using it again
      VK_CHECK_RESULT(vkWaitForFences(
          device, 1, &compute_.fences[currentBuffer], VK_TRUE, UINT64_MAX));
      VK_CHECK_RESULT(
          vkResetFences(device, 1, &compute_.fences[currentBuffer]));
      buildComputeCommandBuffer();

      VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
      computeSubmitInfo.commandBufferCount = 1;
      computeSubmitInfo.pCommandBuffers =
          &compute_.commandBuffers[currentBuffer];
      VK_CHECK_RESULT(vkQueueSubmit(compute_.queue, 1, &computeSubmitInfo,
                                    compute_.fences[currentBuffer]));
    }
    {
      // Graphics
      VulkanExampleBase::prepareFrame();
      updateUniformBuffers();
      buildCommandBuffer();
      VulkanExampleBase::submitFrame();
    }
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
    camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
    camera.setTranslation(glm::vec3(68.0f, 50.0f, 4.0f));
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

  ~VulkanExample() override {
    if (device) {
      // Pipelines
      vkDestroyPipeline(device, graphics_.pipelines.sky_box, nullptr);
      vkDestroyPipeline(device, graphics_.pipelines.wave, nullptr);
      vkDestroyPipeline(device, compute_.pipelines.compose, nullptr);

      // Pipeline Layouts
      vkDestroyPipelineLayout(device, graphics_.pipeline_layouts.sky_box,
                              nullptr);
      vkDestroyPipelineLayout(device, graphics_.pipeline_layouts.wave, nullptr);
      vkDestroyPipelineLayout(device, compute_.pipeline_layouts.compose,
                              nullptr);

      // Buffers: Graphics
      for (auto& buffer : graphics_.uniform_buffers) {
        buffer.sky_box.destroy();
        buffer.wave.destroy();
        buffer.wave_params.destroy();
        buffer.tess_config.destroy();
      }
      // Buffers: Compose
      for (auto& buffer : compute_.uniform_buffers) {
        buffer.compose.destroy();
      }

      // Vertex buffers
      graphics_.wave_mesh_buffers.vertex_buffer.destroy();
      graphics_.wave_mesh_buffers.index_buffer.destroy();

      // Descriptor Layouts
      vkDestroyDescriptorSetLayout(
          device, graphics_.descriptor_set_layouts.sky_box, nullptr);
      vkDestroyDescriptorSetLayout(
          device, graphics_.descriptor_set_layouts.wave, nullptr);
      vkDestroyDescriptorSetLayout(
          device, compute_.descriptor_set_layouts.compose, nullptr);

      // Cube map
      graphics_.textures.cube_map.destroy();

      // Compute textures
      vkDestroyImageView(device, compute_.wave_normal_map.view, nullptr);
      vkDestroyImage(device, compute_.wave_normal_map.image, nullptr);
      vkDestroySampler(device, compute_.wave_normal_map.sampler, nullptr);
      vkFreeMemory(device, compute_.wave_normal_map.deviceMemory, nullptr);

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
