/*
 * Vulkan Example - Smoke simulation with compute shaders and cube marching
 *
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#include <chrono>

#include "VulkanglTFModel.h"
#include "vulkanexamplebase.h"

#define VECTOR_FIELD_FORMAT VK_FORMAT_R32G32B32A32_SFLOAT
#define SCALAR_FIELD_FORMAT VK_FORMAT_R32_SFLOAT

// Vertex layout for this example
struct Vertex {
  float pos[3];
  float uvw[3];
};

struct UiFeatures {
  // emission
  float radius{.25f};

  // Vorticity confinement
  float vorticityStrength{0.12f};

  // Boundary
  int useNoSlip{1};  // 0=free-slip, 1=no-slip
  int jacobiIterationCount{20};

  int timeStep{30};

  // Texture index mappings
  // 0 velocity
  // 1 pressure
  // 4 density
  int textureRadioId{4};

  bool toggleRotation{false};
} uiFeatures;

class VulkanExample : public VulkanExampleBase {
 public:
  // Enable Vulkan 1.3
  VkPhysicalDeviceVulkan13Features enabledFeatures13_{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceVulkan12Features enabledFeatures12_{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};

  // Debug labeling ext
  static constexpr std::array<float, 4> debugColor_ = {.7f, 0.4f, 0.4f, 1.0f};
  static constexpr std::array<float, 4> swapColor_{1, 1, 1, 1};
  PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT{nullptr};
  PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT{nullptr};

  // Handles all compute pipelines
  struct Compute {
    static constexpr int COMPUTE_TEXTURE_DIMENSIONS = 256;
    static constexpr int WORKGROUP_SIZE = 8;

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
    std::array<VkCommandBuffer, MAX_CONCURRENT_FRAMES> commandBuffers{};
    // Fences to make sure command buffers are done
    std::array<VkFence, MAX_CONCURRENT_FRAMES> fences{};

    // Semaphores for submission ordering
    struct ComputeSemaphores {
      VkSemaphore ready{VK_NULL_HANDLE};
      VkSemaphore complete{VK_NULL_HANDLE};
    };
    std::array<ComputeSemaphores, MAX_CONCURRENT_FRAMES> semaphores{};

    // Contains all Vulkan objects that are required to store and use a 3D
    // texture
    struct Texture3D {
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

    // 3D textures needed to store vector/scalar states
    // Split into read/write respectively
    static constexpr int texture_count = 6;
    // Texture index mappings
    // 0 velocity
    // 1 pressure
    // 2 divergence
    // 3 vorticity
    // 4 density
    // 5 temperature
    std::array<Texture3D, texture_count> read_textures;
    std::array<Texture3D, texture_count> write_textures;

    struct EmissionUBO {
      alignas(16) glm::ivec3 gridSize{COMPUTE_TEXTURE_DIMENSIONS};
      alignas(16) glm::vec3 sourceCenter{COMPUTE_TEXTURE_DIMENSIONS / 2.0f,
                                         COMPUTE_TEXTURE_DIMENSIONS / 10.0f,
                                         COMPUTE_TEXTURE_DIMENSIONS / 2.0f};
      alignas(4) float sourceRadius{uiFeatures.radius};
      alignas(4) float deltaTime{1.f / uiFeatures.timeStep};
      alignas(4) float time{0};
    };

    struct AdvectionUBO {
      alignas(16) glm::ivec3 gridSize{COMPUTE_TEXTURE_DIMENSIONS};
      alignas(16) glm::vec3 invGridSize{1.f / COMPUTE_TEXTURE_DIMENSIONS};
      alignas(4) float deltaTime{1.f / uiFeatures.timeStep};
      alignas(4) float dissipation{0.0f};
    };

    struct BuoyancyUBO {
      alignas(16) glm::ivec3 gridSize{COMPUTE_TEXTURE_DIMENSIONS};
      alignas(4) float deltaTime{1.f / uiFeatures.timeStep};
      alignas(4) float buoyancy{0.5f};
      alignas(4) float ambientTemp{0.f};
    };

    struct VorticityUBO {
      alignas(16) glm::ivec3 gridSize{COMPUTE_TEXTURE_DIMENSIONS};
    };

    struct VortConfinementUBO {
      alignas(16) glm::ivec3 gridSize{COMPUTE_TEXTURE_DIMENSIONS};
      alignas(4) float deltaTime{1.f / uiFeatures.timeStep};
      alignas(4) float vorticityStrength{uiFeatures.vorticityStrength};
    };

    struct DivergenceUBO {
      alignas(16) glm::ivec3 gridSize{COMPUTE_TEXTURE_DIMENSIONS};
    };

    struct JacobiUBO {
      alignas(16) glm::ivec3 gridSize{COMPUTE_TEXTURE_DIMENSIONS};
    };

    struct GradientUBO {
      alignas(16) glm::ivec3 gridSize{COMPUTE_TEXTURE_DIMENSIONS};
    };

    struct BoundaryUBO {
      alignas(16) glm::ivec3 gridSize{COMPUTE_TEXTURE_DIMENSIONS};
      alignas(16) glm::vec3 invGridSize{1.f / COMPUTE_TEXTURE_DIMENSIONS};
      // {-X, +X, -Y, +Y,-Z, +Z}  0 = solid, 1 = open
      alignas(16) uint32_t boundaryTypes[6] = {1, 1, 1, 1, 1, 1};
      // 0=free-slip, 1=no-slip
      alignas(16) int useNoSlip{uiFeatures.useNoSlip};
    };

    struct BoundaryPushConstants {
      uint32_t texture_id{};
      int32_t allTextures{0};
    } boundaryPC;

    struct {
      EmissionUBO emission;
      BuoyancyUBO buoyancy;
      AdvectionUBO advection;
      VorticityUBO vorticity;
      VortConfinementUBO vortConfinement;
      DivergenceUBO divergence;
      JacobiUBO jacobi;
      GradientUBO gradient;
      BoundaryUBO boundary;
    } ubos_;

    struct UniformBuffers {
      vks::Buffer emission;
      vks::Buffer buoyancy;
      vks::Buffer advection;
      vks::Buffer vorticity;
      vks::Buffer vortConfinement;
      vks::Buffer divergence;
      vks::Buffer jacobi;
      vks::Buffer gradient;
      vks::Buffer boundary;
    };
    std::array<UniformBuffers, MAX_CONCURRENT_FRAMES> uniformBuffers_;

    // Buffers
    std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers;

    // Pipelines
    struct {
      VkPipeline emission;
      VkPipeline buoyancy;
      VkPipeline advection;
      VkPipeline vorticity;
      VkPipeline vortConfinement;
      VkPipeline divergence;
      VkPipeline jacobi;
      VkPipeline gradient;
      VkPipeline boundary;
    } pipelines_{};
    struct {
      VkPipelineLayout emission{VK_NULL_HANDLE};
      VkPipelineLayout buoyancy{VK_NULL_HANDLE};
      VkPipelineLayout advection{VK_NULL_HANDLE};
      VkPipelineLayout vorticity{VK_NULL_HANDLE};
      VkPipelineLayout vortConfinement{VK_NULL_HANDLE};
      VkPipelineLayout divergence{VK_NULL_HANDLE};
      VkPipelineLayout jacobi{VK_NULL_HANDLE};
      VkPipelineLayout gradient{VK_NULL_HANDLE};
      VkPipelineLayout boundary{VK_NULL_HANDLE};
    } pipelineLayouts_;

    // Descriptors
    struct {
      VkDescriptorSetLayout emission{VK_NULL_HANDLE};
      VkDescriptorSetLayout buoyancy{VK_NULL_HANDLE};
      VkDescriptorSetLayout advection{VK_NULL_HANDLE};
      VkDescriptorSetLayout vorticity{VK_NULL_HANDLE};
      VkDescriptorSetLayout vortConfinement{VK_NULL_HANDLE};
      VkDescriptorSetLayout divergence{VK_NULL_HANDLE};
      VkDescriptorSetLayout jacobi{VK_NULL_HANDLE};
      VkDescriptorSetLayout gradient{VK_NULL_HANDLE};
      VkDescriptorSetLayout boundary{VK_NULL_HANDLE};
    } descriptorSetLayouts_;
    struct DescriptorSets {
      VkDescriptorSet emission{VK_NULL_HANDLE};
      VkDescriptorSet buoyancy{VK_NULL_HANDLE};
      VkDescriptorSet advection{VK_NULL_HANDLE};
      VkDescriptorSet vorticity{VK_NULL_HANDLE};
      VkDescriptorSet vortConfinement{VK_NULL_HANDLE};
      VkDescriptorSet divergence{VK_NULL_HANDLE};
      VkDescriptorSet jacobi{VK_NULL_HANDLE};
      VkDescriptorSet gradient{VK_NULL_HANDLE};
      VkDescriptorSet boundary{VK_NULL_HANDLE};
    };
    std::array<DescriptorSets, MAX_CONCURRENT_FRAMES> descriptorSets_{};
  } compute_;

  // Handles rendering the compute outputs
  struct Graphics {
    static constexpr float CUBE_SCALE = 20.f;
    // families differ and require additional barriers
    uint32_t queueFamilyIndex{0};

    vks::Buffer cubeVerticesBuffer;
    vks::Buffer cubeIndicesBuffer;
    uint32_t indexCount{0};

    std::vector<std::string> viewNames{"Smoke", "Noise", "Entry Rays",
                                       "Exit Rays"};
    struct PreMarchPushConstants {
      // 1 = back faces, 0 = front faces
      alignas(4) uint32_t renderBackFaces{0};
    } preMarchPC;

    struct PreMarchUBO {
      alignas(16) glm::mat4 model;
      alignas(16) glm::mat4 worldViewProjection;
      alignas(16) glm::mat4 invWorldViewProjection;
      alignas(16) glm::vec3 cameraPos;
    };

    struct RayMarchUBO {
      alignas(16) glm::mat4 cameraView;
      alignas(16) glm::mat4 perspective;
      alignas(16) glm::vec3 cameraPos;
      alignas(8) glm::vec2 screenRes;
      alignas(4) float time{0};
      alignas(4) int toggleView{0};  // 0 == 3D texture, 1 == noise, 2 =
                                     // entry Ray, 3 = exit ray
      alignas(4) uint32_t texId{};
    };

    struct UBO {
      PreMarchUBO preMarch;
      RayMarchUBO march;
    } ubos_;

    // vk Buffers
    struct UniformBuffers {
      vks::Buffer preMarch;
      vks::Buffer march;
    };
    std::array<UniformBuffers, MAX_CONCURRENT_FRAMES> uniformBuffers_;

    // Pipelines
    struct {
      VkPipeline preMarch{VK_NULL_HANDLE};
      VkPipeline rayMarch{VK_NULL_HANDLE};
    } pipelines_{};
    struct {
      VkPipelineLayout preMarch{VK_NULL_HANDLE};
      VkPipelineLayout rayMarch{VK_NULL_HANDLE};
    } pipelineLayouts_;

    // Descriptors
    struct {
      VkDescriptorSetLayout preMarch{VK_NULL_HANDLE};
      VkDescriptorSetLayout rayMarch{VK_NULL_HANDLE};
    } descriptorSetLayouts_;
    struct DescriptorSets {
      VkDescriptorSet preMarch{VK_NULL_HANDLE};
      VkDescriptorSet rayMarch{VK_NULL_HANDLE};
    };
    std::array<DescriptorSets, MAX_CONCURRENT_FRAMES> descriptorSets_{};

    // Structure to hold offscreen velocity buffer
    struct VelocityFieldBuffer {
      VkImage image;
      VkDeviceMemory memory;
      VkImageView imageView;
      VkDescriptorImageInfo descriptor;
      VkFormat format;
      VkExtent2D extent;
    };
    struct PreMarchPass {
      VelocityFieldBuffer incoming;
      VelocityFieldBuffer outgoing;
      VkSampler sampler;
    } preMarchPass_{};

  } graphics_;

  // Create a 4 channel 16-bit float velocity buffer
  void createVelocityFieldBuffer(Graphics::VelocityFieldBuffer& buffer) const {
    // Note: VK_FORMAT_R16G16B16_SFLOAT has limited support, so we use RGBA16
    // The alpha channel will just be unused
    buffer.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    buffer.extent = {width_, height_};

    // Create image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = buffer.format;
    imageInfo.extent = {width_, height_, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT |  // For reading as texture
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;  // For copying/readback
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vkCreateImage(device_, &imageInfo, nullptr, &buffer.image);

    // Allocate device_-local memory
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device_, buffer.image, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;

    // Find device_-local memory type
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
      if ((memReqs.memoryTypeBits & (1 << i)) &&
          (memProps.memoryTypes[i].propertyFlags &
           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
        allocInfo.memoryTypeIndex = i;
        break;
      }
    }

    vkAllocateMemory(device_, &allocInfo, nullptr, &buffer.memory);
    vkBindImageMemory(device_, buffer.image, buffer.memory, 0);

    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = buffer.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = buffer.format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    vkCreateImageView(device_, &viewInfo, nullptr, &buffer.imageView);

    // descriptor
    buffer.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    buffer.descriptor.imageView = buffer.imageView;
    buffer.descriptor.sampler = graphics_.preMarchPass_.sampler;
  }

  void preparePreMarchPass() {
    // Create sampler to sample from the textures
    VkSamplerCreateInfo samplerCI = vks::initializers::samplerCreateInfo();
    samplerCI.magFilter = VK_FILTER_LINEAR;
    samplerCI.minFilter = VK_FILTER_LINEAR;
    samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = samplerCI.addressModeU;
    samplerCI.addressModeW = samplerCI.addressModeU;
    samplerCI.mipLodBias = 0.0f;
    samplerCI.maxAnisotropy = 1.0f;
    samplerCI.minLod = 0.0f;
    samplerCI.maxLod = 1.0f;
    samplerCI.unnormalizedCoordinates = VK_FALSE;
    samplerCI.compareEnable = VK_FALSE;
    samplerCI.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    VK_CHECK_RESULT(vkCreateSampler(device_, &samplerCI, nullptr,
                                    &graphics_.preMarchPass_.sampler));

    createVelocityFieldBuffer(graphics_.preMarchPass_.incoming);
    createVelocityFieldBuffer(graphics_.preMarchPass_.outgoing);
  }

  void prepareComputeTexture(Compute::Texture3D& texture,
                             bool readOnly,
                             VkFormat texture_format) const {
    // A 3D texture is described as width x height x depth
    texture.width = compute_.COMPUTE_TEXTURE_DIMENSIONS;
    texture.height = compute_.COMPUTE_TEXTURE_DIMENSIONS;
    texture.depth = compute_.COMPUTE_TEXTURE_DIMENSIONS;
    texture.mipLevels = 1;
    texture.format = texture_format;

    // Format support check
    // 3D texture support in Vulkan is mandatory (in contrast to OpenGL) so no
    // need to check if it's supported
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice_, texture.format,
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
    imageCreateInfo.imageType = VK_IMAGE_TYPE_3D;
    imageCreateInfo.format = texture.format;
    imageCreateInfo.mipLevels = texture.mipLevels;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.extent.width = texture.width;
    imageCreateInfo.extent.height = texture.height;
    imageCreateInfo.extent.depth = texture.depth;
    // Set initial layout of the image to undefined
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_STORAGE_BIT;

    VK_CHECK_RESULT(
        vkCreateImage(device_, &imageCreateInfo, nullptr, &texture.image));

    // Device local memory to back up image
    VkMemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs = {};
    vkGetImageMemoryRequirements(device_, texture.image, &memReqs);
    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex = vulkanDevice_->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device_, &memAllocInfo, nullptr,
                                     &texture.deviceMemory));
    VK_CHECK_RESULT(
        vkBindImageMemory(device_, texture.image, texture.deviceMemory, 0));

    // Transition read textures
    VkCommandBuffer layoutCmd = vulkanDevice_->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    texture.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    vks::tools::setImageLayout(layoutCmd, texture.image,
                               VK_IMAGE_ASPECT_COLOR_BIT,
                               VK_IMAGE_LAYOUT_UNDEFINED, texture.imageLayout);
    if (vulkanDevice_->queueFamilyIndices.graphics !=
        vulkanDevice_->queueFamilyIndices.compute) {
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
          vulkanDevice_->queueFamilyIndices.graphics;
      imageMemoryBarrier.dstQueueFamilyIndex =
          vulkanDevice_->queueFamilyIndices.compute;
      vkCmdPipelineBarrier(layoutCmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_FLAGS_NONE,
                           0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
    }

    vulkanDevice_->flushCommandBuffer(layoutCmd, queue_, true);

    if (readOnly) {
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
          vkCreateSampler(device_, &sampler, nullptr, &texture.sampler));
    }

    // Create image view
    VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
    view.image = texture.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_3D;
    view.format = texture.format;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.baseMipLevel = 0;
    view.subresourceRange.baseArrayLayer = 0;
    view.subresourceRange.layerCount = 1;
    view.subresourceRange.levelCount = 1;
    VK_CHECK_RESULT(vkCreateImageView(device_, &view, nullptr, &texture.view));

    texture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // Fill image descriptor image info to be used descriptor set setup
    if (readOnly) {
      texture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      texture.descriptor.imageView = texture.view;
      texture.descriptor.sampler = texture.sampler;
    }
  }

  // creates an indexed descriptor for all compute textures
  void createTexturesForDescriptorIndexing() {
    // (1) Velocity
    for (int i = 0; i < 1; i++) {
      prepareComputeTexture(compute_.read_textures[i], true,
                            VECTOR_FIELD_FORMAT);
      prepareComputeTexture(compute_.write_textures[i], false,
                            VECTOR_FIELD_FORMAT);
    }
    // (2) Rest are scalar fields
    for (int i = 1; i < compute_.texture_count; i++) {
      prepareComputeTexture(compute_.read_textures[i], true,
                            SCALAR_FIELD_FORMAT);
      prepareComputeTexture(compute_.write_textures[i], false,
                            SCALAR_FIELD_FORMAT);
    }
  }

  void clearAllComputeTextures() const {
    // Clear all textures
    VkCommandBuffer clearCmd = vulkanDevice_->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    VkClearColorValue clearColor = {{0.0f, 0.0f, 0.0f, 0.0f}};
    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = 0;
    range.levelCount = 1;
    range.baseArrayLayer = 0;
    range.layerCount = 1;
    for (int i = 0; i < compute_.texture_count; i++) {
      vkCmdClearColorImage(clearCmd, compute_.read_textures[i].image,
                           VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);
      vkCmdClearColorImage(clearCmd, compute_.write_textures[i].image,
                           VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);
    }
    vulkanDevice_->flushCommandBuffer(clearCmd, queue_, true);
  }

  void prepareComputeTextures() {
    // Create a compute capable device queue
    vkGetDeviceQueue(device_, compute_.queueFamilyIndex, 0, &compute_.queue);

    createTexturesForDescriptorIndexing();
    clearAllComputeTextures();
  }

  void prepareComputeUniformBuffers() {
    for (auto& buffer : compute_.uniformBuffers_) {
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.emission, sizeof(Compute::EmissionUBO),
          &compute_.ubos_.emission));
      VK_CHECK_RESULT(buffer.emission.map());

      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.buoyancy, sizeof(Compute::BuoyancyUBO),
          &compute_.ubos_.buoyancy));
      VK_CHECK_RESULT(buffer.buoyancy.map());

      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.vorticity, sizeof(Compute::VorticityUBO),
          &compute_.ubos_.vorticity));
      VK_CHECK_RESULT(buffer.vorticity.map());

      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.vortConfinement, sizeof(Compute::VortConfinementUBO),
          &compute_.ubos_.vortConfinement));
      VK_CHECK_RESULT(buffer.vortConfinement.map());

      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.advection, sizeof(Compute::AdvectionUBO),
          &compute_.ubos_.advection));
      VK_CHECK_RESULT(buffer.advection.map());

      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.divergence, sizeof(Compute::DivergenceUBO),
          &compute_.ubos_.divergence));
      VK_CHECK_RESULT(buffer.divergence.map());

      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.jacobi, sizeof(Compute::JacobiUBO), &compute_.ubos_.jacobi));
      VK_CHECK_RESULT(buffer.jacobi.map());

      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.gradient, sizeof(Compute::GradientUBO),
          &compute_.ubos_.gradient));
      VK_CHECK_RESULT(buffer.gradient.map());

      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.boundary, sizeof(Compute::BoundaryUBO),
          &compute_.ubos_.boundary));
      VK_CHECK_RESULT(buffer.boundary.map());
    }
  }

  void prepareComputeDescriptors() {
    // Advection
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0 : Array of INPUT textures to advect
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_COMPUTE_BIT, /*binding id*/ 0,
            static_cast<uint32_t>(compute_.read_textures.size())),
        // Binding 1 : Array of OUTPUT textures to write result
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 1,
            static_cast<uint32_t>(compute_.write_textures.size())),
        // Binding 2 : Uniform Buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 2, 1)};
    VkDescriptorSetLayoutCreateInfo descriptorLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);

    // Flags for descriptor arrays - typically want partially bound
    std::vector<VkDescriptorBindingFlags> bindingFlags = {
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,

        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT};

    descriptorLayoutCI.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorLayoutCI.pNext = nullptr;
    descriptorLayoutCI.flags =
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    descriptorLayoutCI.bindingCount =
        static_cast<uint32_t>(setLayoutBindings.size());
    descriptorLayoutCI.pBindings = setLayoutBindings.data();
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorLayoutCI, nullptr,
                                    &compute_.descriptorSetLayouts_.advection));

    // Emission
    setLayoutBindings = {
        // Binding 0 : Array of read-only texs
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_COMPUTE_BIT, /*binding id*/ 0,
            static_cast<uint32_t>(compute_.read_textures.size())),
        // Binding 1 : Array of write-only textures to write result
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 1,
            static_cast<uint32_t>(compute_.write_textures.size())),
        // Binding 2 : Uniform Buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 2, 1)};
    descriptorLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorLayoutCI, nullptr,
                                    &compute_.descriptorSetLayouts_.emission));

    // Buoyancy
    setLayoutBindings = {
        // Binding 0 : Array of read-only texs
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_COMPUTE_BIT, /*binding id*/ 0,
            static_cast<uint32_t>(compute_.read_textures.size())),
        // Binding 1 : Array of write-only textures to write result
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 1,
            static_cast<uint32_t>(compute_.write_textures.size())),
        // Binding 2 : Uniform Buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 2, 1)};
    descriptorLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorLayoutCI, nullptr,
                                    &compute_.descriptorSetLayouts_.buoyancy));

    // Vorticity
    setLayoutBindings = {
        // Binding 0 : Array of read-only texs
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_COMPUTE_BIT, /*binding id*/ 0,
            static_cast<uint32_t>(compute_.read_textures.size())),
        // Binding 1 : Array of write-only textures to write result
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 1,
            static_cast<uint32_t>(compute_.write_textures.size())),
        // Binding 2 : Single velocity field texture
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 2, 1)};
    descriptorLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorLayoutCI, nullptr,
                                    &compute_.descriptorSetLayouts_.vorticity));

    // Vorticity Confinement
    setLayoutBindings = {
        // Binding 0 : Array of read-only texs
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_COMPUTE_BIT, /*binding id*/ 0,
            static_cast<uint32_t>(compute_.read_textures.size())),
        // Binding 1 : Array of write-only textures to write result
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 1,
            static_cast<uint32_t>(compute_.write_textures.size())),
        // Binding 2 : Uniform Buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 2, 1)};
    descriptorLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device_, &descriptorLayoutCI, nullptr,
        &compute_.descriptorSetLayouts_.vortConfinement));

    // Divergence
    setLayoutBindings = {
        // Binding 0 : Array of read-only texs
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_COMPUTE_BIT, /*binding id*/ 0,
            static_cast<uint32_t>(compute_.read_textures.size())),
        // Binding 1 : Array of write-only textures to write result
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 1,
            static_cast<uint32_t>(compute_.write_textures.size())),
        // Binding 2 : Uniform Buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 2, 1)};
    descriptorLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device_, &descriptorLayoutCI, nullptr,
        &compute_.descriptorSetLayouts_.divergence));

    // Jacobi
    setLayoutBindings = {
        // Binding 0 : Array of read-only texs
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_COMPUTE_BIT, /*binding id*/ 0,
            static_cast<uint32_t>(compute_.read_textures.size())),
        // Binding 1 : Array of write-only textures to write result
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 1,
            static_cast<uint32_t>(compute_.write_textures.size())),
        // Binding 2 : Uniform Buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 2, 1)};
    descriptorLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorLayoutCI, nullptr,
                                    &compute_.descriptorSetLayouts_.jacobi));

    // Gradient
    setLayoutBindings = {
        // Binding 0 : Array of read-only texs
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_COMPUTE_BIT, /*binding id*/ 0,
            static_cast<uint32_t>(compute_.read_textures.size())),
        // Binding 1 : Array of write-only textures to write result
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 1,
            static_cast<uint32_t>(compute_.write_textures.size())),
        // Binding 2 : Uniform Buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 2, 1)};
    descriptorLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorLayoutCI, nullptr,
                                    &compute_.descriptorSetLayouts_.gradient));

    // Boundary
    setLayoutBindings = {
        // Binding 0 : Array of read-only texs
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_COMPUTE_BIT, /*binding id*/ 0,
            static_cast<uint32_t>(compute_.read_textures.size())),
        // Binding 1 : Array of write-only textures to write result
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 1,
            static_cast<uint32_t>(compute_.write_textures.size())),
        // Binding 2 : Uniform Buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 2, 1)};
    descriptorLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorLayoutCI, nullptr,
                                    &compute_.descriptorSetLayouts_.boundary));

    // Image descriptors for the 3D texture array
    std::vector<VkDescriptorImageInfo> readOnlyTextureDescriptors(
        compute_.read_textures.size());
    std::vector<VkDescriptorImageInfo> writeOnlyTextureDescriptors(
        compute_.write_textures.size());
    for (size_t i = 0; i < compute_.texture_count; i++) {
      readOnlyTextureDescriptors[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      readOnlyTextureDescriptors[i].sampler = compute_.read_textures[i].sampler;
      readOnlyTextureDescriptors[i].imageView = compute_.read_textures[i].view;

      writeOnlyTextureDescriptors[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      writeOnlyTextureDescriptors[i].sampler = VK_NULL_HANDLE;
      writeOnlyTextureDescriptors[i].imageView =
          compute_.write_textures[i].view;
    }
    // Texture array descriptor
    VkWriteDescriptorSet readOnlyTextureArrayDescriptor = {};
    readOnlyTextureArrayDescriptor.sType =
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    readOnlyTextureArrayDescriptor.dstBinding = 0;
    readOnlyTextureArrayDescriptor.dstArrayElement = 0;
    readOnlyTextureArrayDescriptor.descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    readOnlyTextureArrayDescriptor.descriptorCount =
        static_cast<uint32_t>(compute_.read_textures.size());
    readOnlyTextureArrayDescriptor.pImageInfo =
        readOnlyTextureDescriptors.data();

    VkWriteDescriptorSet writeOnlyTextureArrayDescriptor = {};
    writeOnlyTextureArrayDescriptor.sType =
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeOnlyTextureArrayDescriptor.dstBinding = 1;
    writeOnlyTextureArrayDescriptor.dstArrayElement = 0;
    writeOnlyTextureArrayDescriptor.descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writeOnlyTextureArrayDescriptor.descriptorCount =
        static_cast<uint32_t>(compute_.write_textures.size());
    writeOnlyTextureArrayDescriptor.pImageInfo =
        writeOnlyTextureDescriptors.data();

    // Descriptor set initialization
    for (auto i = 0; i < compute_.uniformBuffers.size(); i++) {
      // Advection
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool_, &compute_.descriptorSetLayouts_.advection, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &compute_.descriptorSets_[i].advection));

      readOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].advection;
      writeOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].advection;

      std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
          readOnlyTextureArrayDescriptor,
          writeOnlyTextureArrayDescriptor,
          vks::initializers::writeDescriptorSet(
              compute_.descriptorSets_[i].advection,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, /*binding id*/ 2,
              &compute_.uniformBuffers_[i].advection.descriptor),
      };
      vkUpdateDescriptorSets(
          device_, static_cast<uint32_t>(computeWriteDescriptorSets.size()),
          computeWriteDescriptorSets.data(), 0, nullptr);

      // Emission
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &compute_.descriptorSetLayouts_.emission, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &compute_.descriptorSets_[i].emission));

      readOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].emission;
      writeOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].emission;
      computeWriteDescriptorSets = {
          readOnlyTextureArrayDescriptor,
          writeOnlyTextureArrayDescriptor,
          vks::initializers::writeDescriptorSet(
              compute_.descriptorSets_[i].emission,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, /*binding id*/ 2,
              &compute_.uniformBuffers_[i].emission.descriptor),

      };
      vkUpdateDescriptorSets(
          device_, static_cast<uint32_t>(computeWriteDescriptorSets.size()),
          computeWriteDescriptorSets.data(), 0, nullptr);
      // Buoyancy
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &compute_.descriptorSetLayouts_.buoyancy, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &compute_.descriptorSets_[i].buoyancy));

      readOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].buoyancy;
      writeOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].buoyancy;
      computeWriteDescriptorSets = {
          readOnlyTextureArrayDescriptor,
          writeOnlyTextureArrayDescriptor,
          vks::initializers::writeDescriptorSet(
              compute_.descriptorSets_[i].buoyancy,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, /*binding id*/ 2,
              &compute_.uniformBuffers_[i].buoyancy.descriptor),

      };
      vkUpdateDescriptorSets(
          device_, static_cast<uint32_t>(computeWriteDescriptorSets.size()),
          computeWriteDescriptorSets.data(), 0, nullptr);

      // Vorticity
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &compute_.descriptorSetLayouts_.vorticity, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &compute_.descriptorSets_[i].vorticity));

      readOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].vorticity;
      writeOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].vorticity;
      computeWriteDescriptorSets = {
          readOnlyTextureArrayDescriptor,
          writeOnlyTextureArrayDescriptor,
          vks::initializers::writeDescriptorSet(
              compute_.descriptorSets_[i].vorticity,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, /*binding id*/ 2,
              &compute_.uniformBuffers_[i].vorticity.descriptor),

      };
      vkUpdateDescriptorSets(
          device_, static_cast<uint32_t>(computeWriteDescriptorSets.size()),
          computeWriteDescriptorSets.data(), 0, nullptr);

      // Vorticity Confinement
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &compute_.descriptorSetLayouts_.vortConfinement, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &compute_.descriptorSets_[i].vortConfinement));

      readOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].vortConfinement;
      writeOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].vortConfinement;
      computeWriteDescriptorSets = {
          readOnlyTextureArrayDescriptor,
          writeOnlyTextureArrayDescriptor,
          vks::initializers::writeDescriptorSet(
              compute_.descriptorSets_[i].vortConfinement,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, /*binding id*/ 2,
              &compute_.uniformBuffers_[i].vortConfinement.descriptor),
      };
      vkUpdateDescriptorSets(
          device_, static_cast<uint32_t>(computeWriteDescriptorSets.size()),
          computeWriteDescriptorSets.data(), 0, nullptr);

      // Divergence
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &compute_.descriptorSetLayouts_.divergence, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &compute_.descriptorSets_[i].divergence));

      readOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].divergence;
      writeOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].divergence;
      computeWriteDescriptorSets = {
          readOnlyTextureArrayDescriptor,
          writeOnlyTextureArrayDescriptor,
          vks::initializers::writeDescriptorSet(
              compute_.descriptorSets_[i].divergence,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, /*binding id*/ 2,
              &compute_.uniformBuffers_[i].divergence.descriptor),
      };
      vkUpdateDescriptorSets(
          device_, static_cast<uint32_t>(computeWriteDescriptorSets.size()),
          computeWriteDescriptorSets.data(), 0, nullptr);

      // Jacobi
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &compute_.descriptorSetLayouts_.jacobi, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &compute_.descriptorSets_[i].jacobi));

      readOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].jacobi;
      writeOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].jacobi;
      computeWriteDescriptorSets = {
          readOnlyTextureArrayDescriptor,
          writeOnlyTextureArrayDescriptor,
          vks::initializers::writeDescriptorSet(
              compute_.descriptorSets_[i].jacobi,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, /*binding id*/ 2,
              &compute_.uniformBuffers_[i].jacobi.descriptor),
      };
      vkUpdateDescriptorSets(
          device_, static_cast<uint32_t>(computeWriteDescriptorSets.size()),
          computeWriteDescriptorSets.data(), 0, nullptr);

      // Gradient
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &compute_.descriptorSetLayouts_.gradient, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &compute_.descriptorSets_[i].gradient));

      readOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].gradient;
      writeOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].gradient;
      computeWriteDescriptorSets = {
          readOnlyTextureArrayDescriptor,
          writeOnlyTextureArrayDescriptor,
          vks::initializers::writeDescriptorSet(
              compute_.descriptorSets_[i].gradient,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, /*binding id*/ 2,
              &compute_.uniformBuffers_[i].gradient.descriptor),
      };
      vkUpdateDescriptorSets(
          device_, static_cast<uint32_t>(computeWriteDescriptorSets.size()),
          computeWriteDescriptorSets.data(), 0, nullptr);

      // Boundary
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &compute_.descriptorSetLayouts_.boundary, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &compute_.descriptorSets_[i].boundary));

      readOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].boundary;
      writeOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].boundary;
      computeWriteDescriptorSets = {
          readOnlyTextureArrayDescriptor,
          writeOnlyTextureArrayDescriptor,
          vks::initializers::writeDescriptorSet(
              compute_.descriptorSets_[i].boundary,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, /*binding id*/ 2,
              &compute_.uniformBuffers_[i].boundary.descriptor),
      };
      vkUpdateDescriptorSets(
          device_, static_cast<uint32_t>(computeWriteDescriptorSets.size()),
          computeWriteDescriptorSets.data(), 0, nullptr);
    }
  }

  void prepareComputePipelines() {
    // Create pipelines
    // Buoyancy
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &compute_.descriptorSetLayouts_.buoyancy, 1);
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr,
                               &compute_.pipelineLayouts_.buoyancy));

    // Emission
    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &compute_.descriptorSetLayouts_.emission, 1);
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr,
                               &compute_.pipelineLayouts_.emission));

    // Advection
    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &compute_.descriptorSetLayouts_.advection, 1);
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr,
                               &compute_.pipelineLayouts_.advection));

    // Vorticity
    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &compute_.descriptorSetLayouts_.vorticity, 1);
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr,
                               &compute_.pipelineLayouts_.vorticity));

    // Vorticity Confinement
    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &compute_.descriptorSetLayouts_.vortConfinement, 1);
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr,
                               &compute_.pipelineLayouts_.vortConfinement));

    // Divergence
    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &compute_.descriptorSetLayouts_.divergence, 1);
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr,
                               &compute_.pipelineLayouts_.divergence));

    // Jacobi
    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &compute_.descriptorSetLayouts_.jacobi, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo,
                                           nullptr,
                                           &compute_.pipelineLayouts_.jacobi));

    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &compute_.descriptorSetLayouts_.gradient, 1);
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr,
                               &compute_.pipelineLayouts_.gradient));

    // Boundary
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(Compute::BoundaryPushConstants);

    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &compute_.descriptorSetLayouts_.boundary, 1);
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr,
                               &compute_.pipelineLayouts_.boundary));

    // Advection
    VkComputePipelineCreateInfo computePipelineCreateInfo =
        vks::initializers::computePipelineCreateInfo(
            compute_.pipelineLayouts_.advection, 0);
    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "smoke/advect.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);

    // We want to use as much shared memory for the compute shader invocations
    // as available, so we calculate it based on the device limits and pass it
    // to the shader via specialization constants
    uint32_t sharedDataSize = std::min(
        static_cast<uint32_t>(1024),
        static_cast<uint32_t>(
            (vulkanDevice_->properties.limits.maxComputeSharedMemorySize /
             sizeof(glm::vec4))));
    VkSpecializationMapEntry specializationMapEntry =
        vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
    VkSpecializationInfo specializationInfo =
        vks::initializers::specializationInfo(1, &specializationMapEntry,
                                              sizeof(int32_t), &sharedDataSize);
    computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;

    VK_CHECK_RESULT(vkCreateComputePipelines(
        device_, pipelineCache_, 1, &computePipelineCreateInfo, nullptr,
        &compute_.pipelines_.advection));

    // Emission
    computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(
        compute_.pipelineLayouts_.emission, 0);
    computePipelineCreateInfo.layout = compute_.pipelineLayouts_.emission;
    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "smoke/emission.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device_, pipelineCache_, 1, &computePipelineCreateInfo, nullptr,
        &compute_.pipelines_.emission));

    // Buoyancy
    computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(
        compute_.pipelineLayouts_.buoyancy, 0);
    computePipelineCreateInfo.layout = compute_.pipelineLayouts_.buoyancy;
    computePipelineCreateInfo.stage = loadShader(
        getShadersPath() + "smoke/buoy.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device_, pipelineCache_, 1, &computePipelineCreateInfo, nullptr,
        &compute_.pipelines_.buoyancy));

    // Vorticity
    computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(
        compute_.pipelineLayouts_.vorticity, 0);
    computePipelineCreateInfo.layout = compute_.pipelineLayouts_.vorticity;
    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "smoke/vorticity.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device_, pipelineCache_, 1, &computePipelineCreateInfo, nullptr,
        &compute_.pipelines_.vorticity));

    // Vorticity Confinment
    computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(
        compute_.pipelineLayouts_.vortConfinement, 0);
    computePipelineCreateInfo.layout =
        compute_.pipelineLayouts_.vortConfinement;
    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "smoke/vortconfinement.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device_, pipelineCache_, 1, &computePipelineCreateInfo, nullptr,
        &compute_.pipelines_.vortConfinement));

    // Divergence
    computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(
        compute_.pipelineLayouts_.divergence, 0);
    computePipelineCreateInfo.layout = compute_.pipelineLayouts_.divergence;
    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "smoke/divergence.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device_, pipelineCache_, 1, &computePipelineCreateInfo, nullptr,
        &compute_.pipelines_.divergence));

    // Jacobi
    computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(
        compute_.pipelineLayouts_.jacobi, 0);
    computePipelineCreateInfo.layout = compute_.pipelineLayouts_.jacobi;
    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "smoke/jacobi.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device_, pipelineCache_, 1, &computePipelineCreateInfo, nullptr,
        &compute_.pipelines_.jacobi));

    // Gradient
    computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(
        compute_.pipelineLayouts_.gradient, 0);
    computePipelineCreateInfo.layout = compute_.pipelineLayouts_.gradient;
    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "smoke/gradient.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device_, pipelineCache_, 1, &computePipelineCreateInfo, nullptr,
        &compute_.pipelines_.gradient));

    // Boundary
    computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(
        compute_.pipelineLayouts_.boundary, 0);
    computePipelineCreateInfo.layout = compute_.pipelineLayouts_.boundary;
    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "smoke/boundary.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device_, pipelineCache_, 1, &computePipelineCreateInfo, nullptr,
        &compute_.pipelines_.boundary));
  }

  void buildComputeCommandBuffer() {
    VkCommandBuffer cmdBuffer = compute_.commandBuffers[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();

    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));
    cmdBeginLabel(cmdBuffer, "Begin Compute Pipelines", {.5f, 0.2f, 3.f, 1.f});

    emissionCmd(cmdBuffer);
    swapTexturesCmd(cmdBuffer);

    // buoyancyCmd(cmdBuffer);
    // swapTexturesCmd(cmdBuffer);

    vorticityCmd(cmdBuffer);
    swapTexturesCmd(cmdBuffer);

    vortConfinementCmd(cmdBuffer);
    swapTexturesCmd(cmdBuffer);

    advectCmd(cmdBuffer);
    swapTexturesCmd(cmdBuffer);

    divergenceCmd(cmdBuffer);

    cmdBeginLabel(cmdBuffer, "Jacobi Iterations Start", {.3f, 0.5f, 0.8f, 1.f});
    for (int i = 0; i < uiFeatures.jacobiIterationCount; i++) {
      std::string text_label = "iteration: " + std::to_string(i);
      cmdBeginLabel(cmdBuffer, text_label.c_str(), {.3f, 0.5f, 0.8f, 1.f});
      jacobiCmd(cmdBuffer);
      cmdEndLabel(cmdBuffer);
      swapTexturesCmd(cmdBuffer);
    }

    gradientCmd(cmdBuffer);
    swapTexturesCmd(cmdBuffer);

    cmdEndLabel(cmdBuffer);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void emissionCmd(const VkCommandBuffer& cmdBuffer) const {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines_.emission);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_.pipelineLayouts_.emission, 0, 1,
                            &compute_.descriptorSets_[currentBuffer_].emission,
                            0, nullptr);
    cmdBeginLabel(cmdBuffer, "Emission", {1.f, .0f, .0f, 1.f});
    vkCmdDispatch(cmdBuffer,
                  compute_.write_textures[0].width / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].height / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].depth / compute_.WORKGROUP_SIZE);
    cmdEndLabel(cmdBuffer);
  }

  void buoyancyCmd(const VkCommandBuffer& cmdBuffer) const {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines_.buoyancy);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_.pipelineLayouts_.buoyancy, 0, 1,
                            &compute_.descriptorSets_[currentBuffer_].buoyancy,
                            0, nullptr);
    cmdBeginLabel(cmdBuffer, "Buoyancy", {1.f, .4f, .4f, 1.f});
    vkCmdDispatch(cmdBuffer,
                  compute_.write_textures[0].width / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].height / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].depth / compute_.WORKGROUP_SIZE);
    cmdEndLabel(cmdBuffer);
  }

  void advectCmd(const VkCommandBuffer& cmdBuffer) const {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines_.advection);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_.pipelineLayouts_.advection, 0, 1,
                            &compute_.descriptorSets_[currentBuffer_].advection,
                            0, nullptr);
    cmdBeginLabel(cmdBuffer, "Advection", {1, 0, 0, 1});
    vkCmdDispatch(cmdBuffer,
                  compute_.write_textures[0].width / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].height / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].depth / compute_.WORKGROUP_SIZE);
    cmdEndLabel(cmdBuffer);
  }

  void vorticityCmd(const VkCommandBuffer& cmdBuffer) const {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines_.vorticity);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_.pipelineLayouts_.vorticity, 0, 1,
                            &compute_.descriptorSets_[currentBuffer_].vorticity,
                            0, nullptr);
    cmdBeginLabel(cmdBuffer, "Vorticity", {1, 1, 0, 1});
    vkCmdDispatch(cmdBuffer,
                  compute_.write_textures[0].width / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].height / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].depth / compute_.WORKGROUP_SIZE);
    cmdEndLabel(cmdBuffer);
  }

  void vortConfinementCmd(const VkCommandBuffer& cmdBuffer) const {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines_.vortConfinement);
    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        compute_.pipelineLayouts_.vortConfinement, 0, 1,
        &compute_.descriptorSets_[currentBuffer_].vortConfinement, 0, nullptr);
    cmdBeginLabel(cmdBuffer, "Vort Confinement", {0, 1, 0, 1});
    vkCmdDispatch(cmdBuffer,
                  compute_.write_textures[0].width / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].height / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].depth / compute_.WORKGROUP_SIZE);
    cmdEndLabel(cmdBuffer);
  }

  void divergenceCmd(const VkCommandBuffer& cmdBuffer) const {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines_.divergence);
    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        compute_.pipelineLayouts_.divergence, 0, 1,
        &compute_.descriptorSets_[currentBuffer_].divergence, 0, nullptr);
    cmdBeginLabel(cmdBuffer, "Divergence", {0, .7f, 0.7f, 1});
    vkCmdDispatch(cmdBuffer,
                  compute_.write_textures[0].width / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].height / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].depth / compute_.WORKGROUP_SIZE);
    cmdEndLabel(cmdBuffer);
  }

  void jacobiCmd(const VkCommandBuffer& cmdBuffer) const {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines_.jacobi);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_.pipelineLayouts_.jacobi, 0, 1,
                            &compute_.descriptorSets_[currentBuffer_].jacobi, 0,
                            nullptr);
    vkCmdDispatch(cmdBuffer,
                  compute_.write_textures[0].width / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].height / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].depth / compute_.WORKGROUP_SIZE);
  }

  void gradientCmd(const VkCommandBuffer& cmdBuffer) const {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines_.gradient);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_.pipelineLayouts_.gradient, 0, 1,
                            &compute_.descriptorSets_[currentBuffer_].gradient,
                            0, nullptr);
    cmdBeginLabel(cmdBuffer, "Gradient", {.5f, .7f, 0.3f, 1.f});
    vkCmdDispatch(cmdBuffer,
                  compute_.write_textures[0].width / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].height / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].depth / compute_.WORKGROUP_SIZE);
    cmdEndLabel(cmdBuffer);
  }

  void boundaryCmd(const VkCommandBuffer& cmdBuffer) {
    boundaryCmd(cmdBuffer, 0, 1);
  }

  void boundaryCmd(const VkCommandBuffer& cmdBuffer,
                   uint32_t textureId,
                   int allTextures = 0) {
    compute_.boundaryPC.texture_id = textureId;
    compute_.boundaryPC.allTextures = allTextures;
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines_.boundary);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_.pipelineLayouts_.boundary, 0, 1,
                            &compute_.descriptorSets_[currentBuffer_].boundary,
                            0, nullptr);
    cmdBeginLabel(cmdBuffer, "Boundary", {.7f, .3f, 0.5f, 1.f});
    vkCmdPushConstants(cmdBuffer, compute_.pipelineLayouts_.boundary,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(Compute::BoundaryPushConstants),
                       &compute_.boundaryPC);

    vkCmdDispatch(cmdBuffer,
                  compute_.write_textures[0].width / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].height / compute_.WORKGROUP_SIZE,
                  compute_.write_textures[0].depth / compute_.WORKGROUP_SIZE);
    cmdEndLabel(cmdBuffer);
  }

  void swapTexturesCmd(const VkCommandBuffer& cmdBuffer) {
    cmdBeginLabel(cmdBuffer, "Swap read/write textures", swapColor_);

    for (int i = 0; i < compute_.texture_count; i++) {
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

      copyRegion.extent.width = compute_.COMPUTE_TEXTURE_DIMENSIONS;
      copyRegion.extent.height = compute_.COMPUTE_TEXTURE_DIMENSIONS;
      copyRegion.extent.depth = compute_.COMPUTE_TEXTURE_DIMENSIONS;

      // Copy output of write to read buffer
      vkCmdCopyImage(cmdBuffer, compute_.write_textures[i].image,
                     VK_IMAGE_LAYOUT_GENERAL, compute_.read_textures[i].image,
                     VK_IMAGE_LAYOUT_GENERAL, 1, &copyRegion);
    }

    cmdEndLabel(cmdBuffer);
  }

  void prepareDescriptorPool() {
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            /*total ubo count */ (/*graphics*/ 2 + /*compute*/ 8) *
                MAX_CONCURRENT_FRAMES),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            /*total texture count (across all pipelines) */ (
                /*graphics: 2 premarch texures + all read textures*/ 2 +
                static_cast<uint32_t>(compute_.read_textures.size()) +
                /*compute textures*/
                static_cast<uint32_t>(compute_.read_textures.size())) *
                MAX_CONCURRENT_FRAMES),
        // textures for writing
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            static_cast<uint32_t>(compute_.write_textures.size() *
                                  MAX_CONCURRENT_FRAMES))};

    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(
            poolSizes,
            /*total descriptor count*/ (/*graphics*/ 2 + /*compute*/ 9) *
                MAX_CONCURRENT_FRAMES);
    // Needed if using VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT in
    // descriptor bindings
    descriptorPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo,
                                           nullptr, &descriptorPool_));
  }

  void prepareCompute() {
    prepareComputeTextures();
    prepareComputeUniformBuffers();
    prepareComputeDescriptors();
    prepareComputePipelines();
    prepareComputeCommandPoolBuffersFencesAndSemaphores();
  }

  void prepareDescriptors() {
    // Layout: Ray march
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0 : Vertex shader uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding_id*/ 0),
        // Binding 1 : Pre march Front
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding_id*/ 1),
        // Binding 2 : Pre march Back
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding_id*/ 2),
        // Binding 3 : Descriptor indexed array of read only textures
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding_id*/ 3, sizeof(compute_.read_textures.size())),
    };

    VkDescriptorSetLayoutCreateInfo descriptorLayout =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorLayout, nullptr,
                                    &graphics_.descriptorSetLayouts_.rayMarch));

    // Layout: Pre march
    setLayoutBindings = {
        // Binding 0 : uniform buffer object
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0)};

    descriptorLayout =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    // Flags for descriptor arrays - typically want partially bound
    std::vector<VkDescriptorBindingFlags> bindingFlags = {
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,

        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT};

    descriptorLayout.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorLayout.pNext = nullptr;
    descriptorLayout.flags =
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    descriptorLayout.bindingCount =
        static_cast<uint32_t>(setLayoutBindings.size());
    descriptorLayout.pBindings = setLayoutBindings.data();

    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorLayout, nullptr,
                                    &graphics_.descriptorSetLayouts_.preMarch));

    // Image descriptors for the 3D texture array
    std::vector<VkDescriptorImageInfo> readOnlyTextureDescriptors(
        compute_.read_textures.size());
    for (size_t i = 0; i < compute_.texture_count; i++) {
      readOnlyTextureDescriptors[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      readOnlyTextureDescriptors[i].sampler = compute_.read_textures[i].sampler;
      readOnlyTextureDescriptors[i].imageView = compute_.read_textures[i].view;
    }
    // Texture array descriptor
    VkWriteDescriptorSet readOnlyTextureArrayDescriptor = {};
    readOnlyTextureArrayDescriptor.sType =
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    /*binding id*/
    readOnlyTextureArrayDescriptor.dstBinding = 3;
    readOnlyTextureArrayDescriptor.dstArrayElement = 0;
    readOnlyTextureArrayDescriptor.descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    readOnlyTextureArrayDescriptor.descriptorCount =
        static_cast<uint32_t>(compute_.read_textures.size());
    readOnlyTextureArrayDescriptor.pImageInfo =
        readOnlyTextureDescriptors.data();

    // Sets per frame, just like the buffers themselves
    // Images do not need to be duplicated per frame, we reuse the same one
    // for each frame
    for (auto i = 0; i < graphics_.uniformBuffers_.size(); i++) {
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool_, &graphics_.descriptorSetLayouts_.rayMarch, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &graphics_.descriptorSets_[i].rayMarch));

      readOnlyTextureArrayDescriptor.dstSet =
          graphics_.descriptorSets_[i].rayMarch;

      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          // Binding 0 : Projection/View matrix as uniform buffer
          vks::initializers::writeDescriptorSet(
              graphics_.descriptorSets_[i].rayMarch,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &graphics_.uniformBuffers_[i].march.descriptor),
          // Binding 1 : Premarch Front
          vks::initializers::writeDescriptorSet(
              graphics_.descriptorSets_[i].rayMarch,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &graphics_.preMarchPass_.incoming.descriptor),
          // Binding 2 : Premarch Front
          vks::initializers::writeDescriptorSet(
              graphics_.descriptorSets_[i].rayMarch,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 2, &graphics_.preMarchPass_.outgoing.descriptor),
          // Binding 3 : Density texture to visualize
          readOnlyTextureArrayDescriptor};
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Premarch
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &graphics_.descriptorSetLayouts_.preMarch, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &graphics_.descriptorSets_[i].preMarch));
      writeDescriptorSets = {
          // Binding 0 : uniform
          vks::initializers::writeDescriptorSet(
              graphics_.descriptorSets_[i].preMarch,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0,
              &graphics_.uniformBuffers_[i].preMarch.descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  void preparePipelines() {
    // Layout: Ray march
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &graphics_.descriptorSetLayouts_.rayMarch, 1);
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr,
                               &graphics_.pipelineLayouts_.rayMarch));

    // Layout: Premarch
    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &graphics_.descriptorSetLayouts_.preMarch, 1);
    // push constants
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(Graphics::PreMarchPushConstants);
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr,
                               &graphics_.pipelineLayouts_.preMarch));

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
            VK_TRUE, /*depth stencil write enabled*/ VK_FALSE,
            VK_COMPARE_OP_LESS_OR_EQUAL);
    VkPipelineViewportStateCreateInfo viewportState =
        vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
    VkPipelineMultisampleStateCreateInfo multisampleState =
        vks::initializers::pipelineMultisampleStateCreateInfo(
            VK_SAMPLE_COUNT_1_BIT, 0);
    std::vector<VkDynamicState> dynamicStateEnables = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_CULL_MODE,
        VK_DYNAMIC_STATE_FRONT_FACE,
    };
    VkPipelineDynamicStateCreateInfo dynamicState =
        vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo =
        vks::initializers::pipelineCreateInfo();

    // Vertex bindings and attributes
    VkVertexInputBindingDescription vertexInputBinding = {
        vks::initializers::vertexInputBindingDescription(
            0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX)};
    std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
        vks::initializers::vertexInputAttributeDescription(
            0, 0, VK_FORMAT_R32G32B32_SFLOAT,
            offsetof(Vertex, pos)),  // Location 0 : Position
        vks::initializers::vertexInputAttributeDescription(
            0, 1, VK_FORMAT_R32G32B32_SFLOAT,
            offsetof(Vertex, uvw)),  // Location 1 : UVW
    };
    VkPipelineVertexInputStateCreateInfo vertexInputStateCI =
        vks::initializers::pipelineVertexInputStateCreateInfo();
    vertexInputStateCI.vertexBindingDescriptionCount = 1;
    vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding;
    vertexInputStateCI.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(vertexInputAttributes.size());
    vertexInputStateCI.pVertexAttributeDescriptions =
        vertexInputAttributes.data();
    pipelineCreateInfo.pVertexInputState = &vertexInputStateCI;

    // Blend state
    blendAttachmentState.colorWriteMask = 0xF;
    blendAttachmentState.blendEnable = VK_FALSE;
    blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachmentState.dstColorBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pColorBlendState = &colorBlendState;
    pipelineCreateInfo.pMultisampleState = &multisampleState;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pDepthStencilState = &depthStencilState;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCreateInfo.pStages = shaderStages.data();

    // New create info to define color, depth and stencil attachments at
    // pipeline create time
    VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo{};
    pipelineRenderingCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    pipelineRenderingCreateInfo.colorAttachmentCount = 1;
    pipelineRenderingCreateInfo.depthAttachmentFormat = depthFormat_;
    pipelineRenderingCreateInfo.stencilAttachmentFormat = depthFormat_;
    // Chain into the pipeline create info
    pipelineCreateInfo.pNext = &pipelineRenderingCreateInfo;

    // Pipeline: Pre march
    pipelineCreateInfo.layout = graphics_.pipelineLayouts_.preMarch;
    shaderStages[0] = loadShader(getShadersPath() + "smoke/premarch.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "smoke/premarch.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &graphics_.preMarchPass_.incoming.format;

    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCreateInfo, nullptr,
                                              &graphics_.pipelines_.preMarch));

    // Pipeline: Ray march
    pipelineCreateInfo.layout = graphics_.pipelineLayouts_.rayMarch;
    shaderStages[0] = loadShader(getShadersPath() + "smoke/raymarch.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "smoke/raymarch.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);

    blendAttachmentState.blendEnable = VK_TRUE;
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &swapChain_.colorFormat_;
    rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;

    depthStencilState.depthTestEnable = VK_FALSE;
    depthStencilState.depthWriteEnable = VK_FALSE;

    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCreateInfo, nullptr,
                                              &graphics_.pipelines_.rayMarch));
  }

  void prepareUniformBuffers() {
    for (auto& buffer : graphics_.uniformBuffers_) {
      // ray march
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.march, sizeof(Graphics::RayMarchUBO),
          &graphics_.ubos_.march));
      VK_CHECK_RESULT(buffer.march.map());

      // pre march
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.preMarch, sizeof(Graphics::PreMarchUBO),
          &graphics_.ubos_.preMarch));
      VK_CHECK_RESULT(buffer.preMarch.map());

      // Init cube state
      auto& model = graphics_.ubos_.preMarch.model;
      model = glm::mat4(1.0f);
      model = glm::scale(model, glm::vec3(graphics_.CUBE_SCALE));
      model = glm::translate(model, glm::vec3(0, 0, 0));
    }
  }

  void updateUniformBuffers() {
    float time =
        std::chrono::duration<float>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();

    // Premarch Uniform
    graphics_.ubos_.preMarch.cameraPos = camera_.position_;
    auto& model = graphics_.ubos_.preMarch.model;
    model = uiFeatures.toggleRotation
                ? glm::rotate(model, glm::radians(float(time) / 10000),
                              glm::vec3(0.f, 1.f, 0.f))
                : model;
    graphics_.ubos_.preMarch.worldViewProjection =
        camera_.matrices_.perspective * camera_.matrices_.view * model;
    graphics_.ubos_.preMarch.invWorldViewProjection =
        glm::inverse(graphics_.ubos_.preMarch.worldViewProjection);
    memcpy(graphics_.uniformBuffers_[currentBuffer_].preMarch.mapped,
           &graphics_.ubos_.preMarch, sizeof(Graphics::PreMarchUBO));

    // Ray March Uniform
    graphics_.ubos_.march.cameraView = camera_.matrices_.view;
    graphics_.ubos_.march.screenRes = glm::vec2(width_, height_);
    graphics_.ubos_.march.perspective = camera_.matrices_.perspective;
    graphics_.ubos_.march.cameraPos = camera_.position_;
    graphics_.ubos_.march.time = time;
    graphics_.ubos_.march.texId = uiFeatures.textureRadioId;
    memcpy(graphics_.uniformBuffers_[currentBuffer_].march.mapped,
           &graphics_.ubos_.march, sizeof(Graphics::RayMarchUBO));

    // Emission
    compute_.ubos_.emission.time = time;
    compute_.ubos_.emission.sourceRadius =
        compute_.COMPUTE_TEXTURE_DIMENSIONS / 2 * uiFeatures.radius;
    compute_.ubos_.emission.deltaTime = 1.f / uiFeatures.timeStep;
    memcpy(compute_.uniformBuffers_[currentBuffer_].emission.mapped,
           &compute_.ubos_.emission, sizeof(Compute::EmissionUBO));

    // Advection
    compute_.ubos_.advection.deltaTime = 1.f / uiFeatures.timeStep;

    // Buoyancy
    compute_.ubos_.buoyancy.deltaTime = 1.f / uiFeatures.timeStep;
    memcpy(compute_.uniformBuffers_[currentBuffer_].buoyancy.mapped,
           &compute_.ubos_.buoyancy, sizeof(Compute::BuoyancyUBO));

    // Vorticity Confinement
    compute_.ubos_.vortConfinement.vorticityStrength =
        uiFeatures.vorticityStrength;
    compute_.ubos_.vortConfinement.deltaTime = 1.f / uiFeatures.timeStep;
    memcpy(compute_.uniformBuffers_[currentBuffer_].vortConfinement.mapped,
           &compute_.ubos_.vortConfinement,
           sizeof(Compute::VortConfinementUBO));

    // Boundary
    compute_.ubos_.boundary.useNoSlip = uiFeatures.useNoSlip;
    memcpy(compute_.uniformBuffers_[currentBuffer_].boundary.mapped,
           &compute_.ubos_.boundary, sizeof(Compute::BoundaryUBO));
  }

  void prepareGraphics() {
    generateCube();
    preparePreMarchPass();
    prepareUniformBuffers();
    prepareDescriptors();
    preparePipelines();
  }

  void prepare() override {
    VulkanExampleBase::prepare();
    graphics_.queueFamilyIndex = vulkanDevice_->queueFamilyIndices.graphics;
    compute_.queueFamilyIndex = vulkanDevice_->queueFamilyIndices.compute;
    prepareDescriptorPool();
    prepareCompute();
    prepareGraphics();
    prepareDebugExt();
    prepared_ = true;
  }

  void render() override {
    if (!prepared_) {
      return;
    }
    {
      // Use a fence to ensure that compute command buffer has finished
      // executing before using it again
      vkWaitForFences(device_, 1, &compute_.fences[currentBuffer_], VK_TRUE,
                      UINT64_MAX);
      vkResetFences(device_, 1, &compute_.fences[currentBuffer_]);

      buildComputeCommandBuffer();

      VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
      computeSubmitInfo.commandBufferCount = 1;
      computeSubmitInfo.pCommandBuffers =
          &compute_.commandBuffers[currentBuffer_];
      VK_CHECK_RESULT(vkQueueSubmit(compute_.queue, 1, &computeSubmitInfo,
                                    compute_.fences[currentBuffer_]));
    }
    {
      VulkanExampleBase::prepareFrame();
      updateUniformBuffers();
      buildGraphicsCommandBuffer();
      VulkanExampleBase::submitFrame();
    }
  }

  void buildGraphicsCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));
    frontPreMarchCmd(cmdBuffer);
    backPreMarchCmd(cmdBuffer);
    rayMarchCmd(cmdBuffer);
    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void frontPreMarchCmd(const VkCommandBuffer& cmdBuffer) {
    // With dynamic rendering there are no subpass dependencies, so we need to
    // take care of proper layout transitions by using barriers This set of
    // barriers prepares the color and depth images for output
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, graphics_.preMarchPass_.incoming.image, 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, depthStencil_.image, 0,
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
    VkRenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachment.imageView = graphics_.preMarchPass_.incoming.imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {0.0f, 0.0f, 0.0f, 0.0f};

    VkRenderingInfoKHR renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    renderingInfo.renderArea = {0, 0, width_, height_};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = nullptr;
    renderingInfo.pStencilAttachment = nullptr;

    vkCmdBeginRendering(cmdBuffer, &renderingInfo);

    // Set viewport and scissor
    VkViewport viewport{0.0f, 0.0f, (float)width_, (float)height_, 0.0f, 1.0f};
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics_.pipelines_.preMarch);
    vkCmdSetCullMode(cmdBuffer, VkCullModeFlagBits(VK_CULL_MODE_BACK_BIT));
    vkCmdSetFrontFace(cmdBuffer, VK_FRONT_FACE_CLOCKWISE);

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            graphics_.pipelineLayouts_.preMarch, 0, 1,
                            &graphics_.descriptorSets_[currentBuffer_].preMarch,
                            0, nullptr);

    VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1,
                           &graphics_.cubeVerticesBuffer.buffer, offsets);
    vkCmdBindIndexBuffer(cmdBuffer, graphics_.cubeIndicesBuffer.buffer, 0,
                         VK_INDEX_TYPE_UINT32);

    cmdBeginLabel(cmdBuffer, "Front Pre Marching", {0.1f, 0.6f, 0.6f, 1.f});
    graphics_.preMarchPC.renderBackFaces = 0;
    vkCmdPushConstants(cmdBuffer, graphics_.pipelineLayouts_.preMarch,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(Graphics::PreMarchPushConstants),
                       &graphics_.preMarchPC);

    vkCmdDrawIndexed(cmdBuffer, graphics_.indexCount, 1, 0, 0, 0);
    cmdEndLabel(cmdBuffer);

    vkCmdEndRendering(cmdBuffer);
  }

  void backPreMarchCmd(const VkCommandBuffer& cmdBuffer) {
    // With dynamic rendering there are no subpass dependencies, so we need to
    // take care of proper layout transitions by using barriers This set of
    // barriers prepares the color and depth images for output
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, graphics_.preMarchPass_.outgoing.image, 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, depthStencil_.image, 0,
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
    VkRenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachment.imageView = graphics_.preMarchPass_.outgoing.imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {0.0f, 0.0f, 0.0f, 0.0f};

    VkRenderingInfoKHR renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    renderingInfo.renderArea = {0, 0, width_, height_};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = nullptr;
    renderingInfo.pStencilAttachment = nullptr;

    vkCmdBeginRendering(cmdBuffer, &renderingInfo);

    // Set viewport and scissor
    VkViewport viewport{0.0f, 0.0f, (float)width_, (float)height_, 0.0f, 1.0f};
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics_.pipelines_.preMarch);
    vkCmdSetCullMode(cmdBuffer, VkCullModeFlagBits(VK_CULL_MODE_FRONT_BIT));
    vkCmdSetFrontFace(cmdBuffer, VK_FRONT_FACE_CLOCKWISE);

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            graphics_.pipelineLayouts_.preMarch, 0, 1,
                            &graphics_.descriptorSets_[currentBuffer_].preMarch,
                            0, nullptr);

    VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1,
                           &graphics_.cubeVerticesBuffer.buffer, offsets);
    vkCmdBindIndexBuffer(cmdBuffer, graphics_.cubeIndicesBuffer.buffer, 0,
                         VK_INDEX_TYPE_UINT32);

    cmdBeginLabel(cmdBuffer, "Back Pre Marching", {0.3f, 0.6f, .1f, 1.f});
    graphics_.preMarchPC.renderBackFaces = 1;
    vkCmdPushConstants(cmdBuffer, graphics_.pipelineLayouts_.preMarch,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(Graphics::PreMarchPushConstants),
                       &graphics_.preMarchPC);
    vkCmdDrawIndexed(cmdBuffer, graphics_.indexCount, 1, 0, 0, 0);
    cmdEndLabel(cmdBuffer);

    vkCmdEndRendering(cmdBuffer);
  }

  void rayMarchCmd(const VkCommandBuffer& cmdBuffer) {
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, swapChain_.images_[currentImageIndex_], 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, depthStencil_.image, 0,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VkImageSubresourceRange{
            VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0,
            1});

    // Need to change the format of the textures before reading
    // -_-
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, graphics_.preMarchPass_.incoming.image, 0,
        VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, graphics_.preMarchPass_.outgoing.image, 0,
        VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    // New structures are used to define the attachments used in dynamic
    // rendering
    VkRenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachment.imageView = swapChain_.imageViews_[currentImageIndex_];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {0.0f, 0.0f, 0.0f, 0.0f};

    // A single depth stencil attachment info can be used, but they can also
    // be specified separately. When both are specified separately, the only
    // requirement is that the image view is identical.
    VkRenderingAttachmentInfoKHR depthStencilAttachment{};
    depthStencilAttachment.sType =
        VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    depthStencilAttachment.imageView = depthStencil_.view;
    depthStencilAttachment.imageLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthStencilAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthStencilAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthStencilAttachment.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfoKHR renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    renderingInfo.renderArea = {0, 0, width_, height_};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthStencilAttachment;
    renderingInfo.pStencilAttachment = &depthStencilAttachment;

    // Begin dynamic rendering
    vkCmdBeginRendering(cmdBuffer, &renderingInfo);
    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            graphics_.pipelineLayouts_.rayMarch, 0, 1,
                            &graphics_.descriptorSets_[currentBuffer_].rayMarch,
                            0, nullptr);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics_.pipelines_.rayMarch);
    vkCmdSetCullMode(cmdBuffer, VkCullModeFlagBits(VK_CULL_MODE_FRONT_BIT));
    vkCmdSetFrontFace(cmdBuffer, VK_FRONT_FACE_CLOCKWISE);

    VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1,
                           &graphics_.cubeVerticesBuffer.buffer, offsets);
    vkCmdBindIndexBuffer(cmdBuffer, graphics_.cubeIndicesBuffer.buffer, 0,
                         VK_INDEX_TYPE_UINT32);

    cmdBeginLabel(cmdBuffer, "Cube Marching");
    vkCmdDrawIndexed(cmdBuffer, graphics_.indexCount, 1, 0, 0, 0);
    cmdEndLabel(cmdBuffer);

    cmdBeginLabel(cmdBuffer, "Draw UI", {0, 0, 0, 1});
    drawUI(cmdBuffer);
    cmdEndLabel(cmdBuffer);

    // End dynamic rendering
    vkCmdEndRendering(cmdBuffer);
  }

  void generateCube() {
    // Setup vertices indices for a cube
    std::vector<Vertex> vertices = {
        {{-0.5, -0.5, -0.5}, {0, 0, 0}},  // 0: back-bottom-left
        {{0.5, -0.5, -0.5}, {1, 0, 0}},   // 1: back-bottom-right
        {{0.5, 0.5, -0.5}, {1, 1, 0}},    // 2: back-top-right
        {{-0.5, 0.5, -0.5}, {0, 1, 0}},   // 3: back-top-left
        {{-0.5, -0.5, 0.5}, {0, 0, 1}},   // 4: front-bottom-left
        {{0.5, -0.5, 0.5}, {1, 0, 1}},    // 5: front-bottom-right
        {{0.5, 0.5, 0.5}, {1, 1, 1}},     // 6: front-top-right
        {{-0.5, 0.5, 0.5}, {0, 1, 1}},    // 7: front-top-left
    };

    std::vector<uint32_t> indices = {
        4, 5, 6, 4, 6, 7,  // Front
        1, 0, 3, 1, 3, 2,  // Back
        7, 6, 2, 7, 2, 3,  // Top
        0, 1, 5, 0, 5, 4,  // Bottom
        5, 1, 2, 5, 2, 6,  // Right
        0, 4, 7, 0, 7, 3,  // Left
    };

    graphics_.indexCount = static_cast<uint32_t>(indices.size());

    // Create buffers
    // For the sake of simplicity we won't stage the vertex data to the gpu
    // memory Vertex buffer
    VK_CHECK_RESULT(vulkanDevice_->createBuffer(
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &graphics_.cubeVerticesBuffer, vertices.size() * sizeof(Vertex),
        vertices.data()));
    // Index buffer
    VK_CHECK_RESULT(vulkanDevice_->createBuffer(
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &graphics_.cubeIndicesBuffer, indices.size() * sizeof(uint32_t),
        indices.data()));
  }

  void prepareComputeCommandPoolBuffersFencesAndSemaphores() {
    // Separate command pool as queue family for compute may be different than
    // graphics
    VkCommandPoolCreateInfo cmdPoolInfo =
        vks::initializers::commandPoolCreateInfo();
    cmdPoolInfo.queueFamilyIndex = compute_.queueFamilyIndex;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK_RESULT(vkCreateCommandPool(device_, &cmdPoolInfo, nullptr,
                                        &compute_.commandPool));

    // Create command buffers for compute operations
    for (auto& cmdBuffer : compute_.commandBuffers) {
      cmdBuffer = vulkanDevice_->createCommandBuffer(
          VK_COMMAND_BUFFER_LEVEL_PRIMARY, compute_.commandPool);
    }

    // Fences to check for command buffer completion
    for (auto& fence : compute_.fences) {
      VkFenceCreateInfo fenceCreateInfo =
          vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
      VK_CHECK_RESULT(
          vkCreateFence(device_, &fenceCreateInfo, nullptr, &fence));
    }

    // Semaphores to order compute and graphics submissions
    for (auto& semaphore : compute_.semaphores) {
      VkSemaphoreCreateInfo semaphoreInfo{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &semaphore.ready);
      vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &semaphore.complete);
    }
    // Signal first used ready semaphore
    VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
    computeSubmitInfo.signalSemaphoreCount = 1;
    computeSubmitInfo.pSignalSemaphores =
        &compute_.semaphores[MAX_CONCURRENT_FRAMES - 1].ready;
    VK_CHECK_RESULT(
        vkQueueSubmit(compute_.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE));
  }

  virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay) override {
    if (overlay->header("Settings")) {
      overlay->comboBox("Select View", &graphics_.ubos_.march.toggleView,
                        graphics_.viewNames);
      if (graphics_.ubos_.march.toggleView == 0) {
        overlay->sliderFloat("Smoke Radius", &uiFeatures.radius, 0, 1);
        overlay->sliderFloat("Vorticity Strength",
                             &uiFeatures.vorticityStrength, 0.0f, 1.f);
        overlay->sliderInt("Jacobi Iterations",
                           &uiFeatures.jacobiIterationCount, 1, 60);
        overlay->sliderInt("1 / Time Step", &uiFeatures.timeStep, 1, 360);

        overlay->radioButton("Smoke Texture", &uiFeatures.textureRadioId, 4);
        overlay->radioButton("Velocity Texture", &uiFeatures.textureRadioId, 0);
        overlay->radioButton("Pressure Texture", &uiFeatures.textureRadioId, 1);
      }

      overlay->checkBox("Toggle Rotation", &uiFeatures.toggleRotation);
      if (overlay->button("Reset")) {
        clearAllComputeTextures();
      }
    }
  }

  void prepareDebugExt() {
    vkCmdBeginDebugUtilsLabelEXT =
        reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
            vkGetInstanceProcAddr(instance_, "vkCmdBeginDebugUtilsLabelEXT"));
    vkCmdEndDebugUtilsLabelEXT =
        reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
            vkGetInstanceProcAddr(instance_, "vkCmdEndDebugUtilsLabelEXT"));
  }

  void cmdBeginLabel(const VkCommandBuffer& command_buffer,
                     const char* label_name,
                     std::array<float, 4> color = debugColor_) const {
    VkDebugUtilsLabelEXT label = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = label_name;
    memcpy(label.color, color.data(), sizeof(float) * 4);
    vkCmdBeginDebugUtilsLabelEXT(command_buffer, &label);
  }

  void cmdEndLabel(const VkCommandBuffer& command_buffer) const {
    vkCmdEndDebugUtilsLabelEXT(command_buffer);
  }

  void setupRenderPass() override {
    // With VK_KHR_dynamic_rendering we no longer need a render pass, so
    // skip the sample base render pass setup
    renderPass_ = VK_NULL_HANDLE;
  }

  void setupFrameBuffer() override {
    // With VK_KHR_dynamic_rendering we no longer need a frame buffer
    // LEAVE THIS EMPTY
  }

  VulkanExample() : VulkanExampleBase() {
    title_ = "Smoke Simulation";
    camera_.type_ = Camera::CameraType::firstperson;
    camera_.setMovementSpeed(25.f);
    camera_.setPosition(glm::vec3(0.0f, 0.0f, -30.f));
    camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);
    width_ = uint32_t(width_ * 1.5f);
    height_ = uint32_t(height_ * 1.5f);

    apiVersion_ = VK_API_VERSION_1_3;

    // Descriptor indexing
    enabledFeatures12_.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    enabledFeatures12_.runtimeDescriptorArray = VK_TRUE;
    enabledFeatures12_.descriptorBindingVariableDescriptorCount = VK_TRUE;

    // Dynamic rendering
    enabledFeatures13_.dynamicRendering = VK_TRUE;
    enabledFeatures13_.pNext = &enabledFeatures12_;

    deviceCreatepNextChain_ = &enabledFeatures13_;
  }

  ~VulkanExample() override {
    if (device_) {
      // Graphics
      vkDestroyPipeline(device_, graphics_.pipelines_.rayMarch, nullptr);
      vkDestroyPipeline(device_, graphics_.pipelines_.preMarch, nullptr);

      vkDestroyPipelineLayout(device_, graphics_.pipelineLayouts_.preMarch,
                              nullptr);
      vkDestroyPipelineLayout(device_, graphics_.pipelineLayouts_.rayMarch,
                              nullptr);

      vkDestroyDescriptorSetLayout(
          device_, graphics_.descriptorSetLayouts_.preMarch, nullptr);
      vkDestroyDescriptorSetLayout(
          device_, graphics_.descriptorSetLayouts_.rayMarch, nullptr);

      for (auto& buffer : graphics_.uniformBuffers_) {
        buffer.preMarch.destroy();
        buffer.march.destroy();
      }
      graphics_.cubeVerticesBuffer.destroy();
      graphics_.cubeIndicesBuffer.destroy();

      // Pre march pass images
      vkDestroyImageView(device_, graphics_.preMarchPass_.incoming.imageView,
                         nullptr);
      vkDestroyImage(device_, graphics_.preMarchPass_.incoming.image, nullptr);
      vkFreeMemory(device_, graphics_.preMarchPass_.incoming.memory, nullptr);

      vkDestroyImageView(device_, graphics_.preMarchPass_.outgoing.imageView,
                         nullptr);
      vkDestroyImage(device_, graphics_.preMarchPass_.outgoing.image, nullptr);
      vkFreeMemory(device_, graphics_.preMarchPass_.outgoing.memory, nullptr);
      vkDestroySampler(device_, graphics_.preMarchPass_.sampler, nullptr);

      // Compute
      vkDestroyPipeline(device_, compute_.pipelines_.emission, nullptr);
      vkDestroyPipeline(device_, compute_.pipelines_.advection, nullptr);
      vkDestroyPipeline(device_, compute_.pipelines_.buoyancy, nullptr);
      vkDestroyPipeline(device_, compute_.pipelines_.vorticity, nullptr);
      vkDestroyPipeline(device_, compute_.pipelines_.vortConfinement, nullptr);
      vkDestroyPipeline(device_, compute_.pipelines_.divergence, nullptr);
      vkDestroyPipeline(device_, compute_.pipelines_.jacobi, nullptr);
      vkDestroyPipeline(device_, compute_.pipelines_.gradient, nullptr);
      vkDestroyPipeline(device_, compute_.pipelines_.boundary, nullptr);

      vkDestroyPipelineLayout(device_, compute_.pipelineLayouts_.emission,
                              nullptr);
      vkDestroyPipelineLayout(device_, compute_.pipelineLayouts_.advection,
                              nullptr);
      vkDestroyPipelineLayout(device_, compute_.pipelineLayouts_.buoyancy,
                              nullptr);
      vkDestroyPipelineLayout(device_, compute_.pipelineLayouts_.vorticity,
                              nullptr);
      vkDestroyPipelineLayout(
          device_, compute_.pipelineLayouts_.vortConfinement, nullptr);
      vkDestroyPipelineLayout(device_, compute_.pipelineLayouts_.divergence,
                              nullptr);
      vkDestroyPipelineLayout(device_, compute_.pipelineLayouts_.jacobi,
                              nullptr);
      vkDestroyPipelineLayout(device_, compute_.pipelineLayouts_.gradient,
                              nullptr);
      vkDestroyPipelineLayout(device_, compute_.pipelineLayouts_.boundary,
                              nullptr);

      // Compute Textures
      for (auto& texture : compute_.read_textures) {
        if (texture.view != VK_NULL_HANDLE) {
          vkDestroyImageView(device_, texture.view, nullptr);
        }
        if (texture.image != VK_NULL_HANDLE) {
          vkDestroyImage(device_, texture.image, nullptr);
        }
        if (texture.sampler != VK_NULL_HANDLE) {
          vkDestroySampler(device_, texture.sampler, nullptr);
        }
        if (texture.deviceMemory != VK_NULL_HANDLE) {
          vkFreeMemory(device_, texture.deviceMemory, nullptr);
        }
      }
      for (auto& texture : compute_.write_textures) {
        if (texture.view != VK_NULL_HANDLE) {
          vkDestroyImageView(device_, texture.view, nullptr);
        }
        if (texture.image != VK_NULL_HANDLE) {
          vkDestroyImage(device_, texture.image, nullptr);
        }
        if (texture.sampler != VK_NULL_HANDLE) {
          vkDestroySampler(device_, texture.sampler, nullptr);
        }
        if (texture.deviceMemory != VK_NULL_HANDLE) {
          vkFreeMemory(device_, texture.deviceMemory, nullptr);
        }
      }

      // Compute Buffers
      for (auto& buffer : compute_.uniformBuffers_) {
        buffer.emission.destroy();
        buffer.buoyancy.destroy();
        buffer.advection.destroy();
        buffer.vorticity.destroy();
        buffer.vortConfinement.destroy();
        buffer.divergence.destroy();
        buffer.jacobi.destroy();
        buffer.gradient.destroy();
        buffer.boundary.destroy();
      }

      vkDestroyDescriptorSetLayout(
          device_, compute_.descriptorSetLayouts_.emission, nullptr);
      vkDestroyDescriptorSetLayout(
          device_, compute_.descriptorSetLayouts_.advection, nullptr);
      vkDestroyDescriptorSetLayout(
          device_, compute_.descriptorSetLayouts_.buoyancy, nullptr);
      vkDestroyDescriptorSetLayout(
          device_, compute_.descriptorSetLayouts_.vorticity, nullptr);
      vkDestroyDescriptorSetLayout(
          device_, compute_.descriptorSetLayouts_.vortConfinement, nullptr);
      vkDestroyDescriptorSetLayout(
          device_, compute_.descriptorSetLayouts_.divergence, nullptr);
      vkDestroyDescriptorSetLayout(
          device_, compute_.descriptorSetLayouts_.jacobi, nullptr);
      vkDestroyDescriptorSetLayout(
          device_, compute_.descriptorSetLayouts_.gradient, nullptr);
      vkDestroyDescriptorSetLayout(
          device_, compute_.descriptorSetLayouts_.boundary, nullptr);

      vkDestroyCommandPool(device_, compute_.commandPool, nullptr);
      for (auto& fence : compute_.fences) {
        vkDestroyFence(device_, fence, nullptr);
      }
      for (auto& semaphore : compute_.semaphores) {
        vkDestroySemaphore(device_, semaphore.ready, nullptr);
        vkDestroySemaphore(device_, semaphore.complete, nullptr);
      }
    }
  }
};

VULKAN_EXAMPLE_MAIN()
