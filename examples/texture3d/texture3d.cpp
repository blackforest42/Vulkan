/*
 * Vulkan Example - 3D texture loading (and generation using perlin noise)
 * example
 *
 * Copyright (C) 2016-2025 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#include "vulkanexamplebase.h"

// Vertex layout for this example
struct Vertex {
  float pos[3];
  float uv[2];
  float normal[3];
};

// Translation of Ken Perlin's JAVA implementation
// (http://mrl.nyu.edu/~perlin/noise/)
template <typename T>
class PerlinNoise {
 private:
  uint32_t permutations[512];
  T fade(T t) { return t * t * t * (t * (t * (T)6 - (T)15) + (T)10); }
  T lerp(T t, T a, T b) { return a + t * (b - a); }
  T grad(int hash, T x, T y, T z) {
    // Convert LO 4 bits of hash code into 12 gradient directions
    int h = hash & 15;
    T u = h < 8 ? x : y;
    T v = h < 4 ? y : h == 12 || h == 14 ? x : z;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
  }

 public:
  PerlinNoise(bool applyRandomSeed) {
    // Generate random lookup for permutations containing all numbers from
    // 0..255
    std::vector<uint8_t> plookup;
    plookup.resize(256);
    std::iota(plookup.begin(), plookup.end(), 0);
    std::default_random_engine rndEngine(
        applyRandomSeed ? std::random_device{}() : 0);
    std::shuffle(plookup.begin(), plookup.end(), rndEngine);

    for (uint32_t i = 0; i < 256; i++) {
      permutations[i] = permutations[256 + i] = plookup[i];
    }
  }
  T noise(T x, T y, T z) {
    // Find unit cube that contains point
    int32_t X = (int32_t)floor(x) & 255;
    int32_t Y = (int32_t)floor(y) & 255;
    int32_t Z = (int32_t)floor(z) & 255;
    // Find relative x,y,z of point in cube
    x -= floor(x);
    y -= floor(y);
    z -= floor(z);

    // Compute fade curves for each of x,y,z
    T u = fade(x);
    T v = fade(y);
    T w = fade(z);

    // Hash coordinates of the 8 cube corners
    uint32_t A = permutations[X] + Y;
    uint32_t AA = permutations[A] + Z;
    uint32_t AB = permutations[A + 1] + Z;
    uint32_t B = permutations[X + 1] + Y;
    uint32_t BA = permutations[B] + Z;
    uint32_t BB = permutations[B + 1] + Z;

    // And add blended results for 8 corners of the cube;
    T res = lerp(w,
                 lerp(v,
                      lerp(u, grad(permutations[AA], x, y, z),
                           grad(permutations[BA], x - 1, y, z)),
                      lerp(u, grad(permutations[AB], x, y - 1, z),
                           grad(permutations[BB], x - 1, y - 1, z))),
                 lerp(v,
                      lerp(u, grad(permutations[AA + 1], x, y, z - 1),
                           grad(permutations[BA + 1], x - 1, y, z - 1)),
                      lerp(u, grad(permutations[AB + 1], x, y - 1, z - 1),
                           grad(permutations[BB + 1], x - 1, y - 1, z - 1))));
    return res;
  }
};

// Fractal noise generator based on perlin noise above
template <typename T>
class FractalNoise {
 private:
  PerlinNoise<T> perlinNoise;
  uint32_t octaves;
  T frequency;
  T amplitude;
  T persistence;

 public:
  FractalNoise(const PerlinNoise<T>& perlinNoiseIn)
      : perlinNoise(perlinNoiseIn) {
    octaves = 6;
    persistence = (T)0.5;
  }

  T noise(T x, T y, T z) {
    T sum = 0;
    T frequency = (T)1;
    T amplitude = (T)1;
    T max = (T)0;
    for (uint32_t i = 0; i < octaves; i++) {
      sum += perlinNoise.noise(x * frequency, y * frequency, z * frequency) *
             amplitude;
      max += amplitude;
      amplitude *= persistence;
      frequency *= (T)2;
    }

    sum = sum / max;
    return (sum + (T)1.0) / (T)2.0;
  }
};

class VulkanExample : public VulkanExampleBase {
 public:
  // Contains all Vulkan objects that are required to store and use a 3D texture
  struct Texture {
    VkSampler sampler = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkImageLayout imageLayout;
    VkDeviceMemory deviceMemory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDescriptorImageInfo descriptor;
    VkFormat format;
    uint32_t width{0};
    uint32_t height{0};
    uint32_t depth{0};
    uint32_t mipLevels{0};
  } texture_;

  vks::Buffer vertexBuffer;
  vks::Buffer indexBuffer;
  uint32_t indexCount{0};

  struct UniformData {
    glm::mat4 projection;
    glm::mat4 modelView;
    glm::vec4 viewPos;
    // The current depth level of the texture to display
    // This is animated
    float depth = 0.0f;
  } uniformData_;
  std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers_;

  VkPipeline pipeline{VK_NULL_HANDLE};
  VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
  VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
  std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSets_{};

  VulkanExample() : VulkanExampleBase() {
    title = "3D textures";
    camera_.type_ = Camera::CameraType::lookat;
    camera_.setPosition(glm::vec3(0.0f, 0.0f, -2.5f));
    camera_.setRotation(glm::vec3(0.0f, 15.0f, 0.0f));
    camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);
    srand(benchmark.active ? 0 : (unsigned int)time(NULL));
  }

  ~VulkanExample() {
    if (device_) {
      destroyTextureImage(texture_);
      vkDestroyPipeline(device_, pipeline, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayout, nullptr);
      vertexBuffer.destroy();
      indexBuffer.destroy();
      for (auto& buffer : uniformBuffers_) {
        buffer.destroy();
      }
    }
  }

  // Prepare all Vulkan resources for the 3D texture (including descriptors)
  // Does not fill the texture with data
  void prepareNoiseTexture(uint32_t width, uint32_t height, uint32_t depth) {
    // A 3D texture is described as width x height x depth
    texture_.width = width;
    texture_.height = height;
    texture_.depth = depth;
    texture_.mipLevels = 1;
    texture_.format = VK_FORMAT_R8_UNORM;

    // Format support check
    // 3D texture support in Vulkan is mandatory (in contrast to OpenGL) so no
    // need to check if it's supported
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice_, texture_.format,
                                        &formatProperties);
    // Check if format supports transfer
    if (!(formatProperties.optimalTilingFeatures &
          VK_FORMAT_FEATURE_TRANSFER_DST_BIT)) {
      std::cout << "Error: Device does not support flag TRANSFER_DST for "
                   "selected texture format!"
                << std::endl;
      return;
    }
    // Check if GPU supports requested 3D texture dimensions
    uint32_t maxImageDimension3D(
        vulkanDevice_->properties.limits.maxImageDimension3D);
    if (width > maxImageDimension3D || height > maxImageDimension3D ||
        depth > maxImageDimension3D) {
      std::cout << "Error: Requested texture dimensions is greater than "
                   "supported 3D texture dimension!"
                << std::endl;
      return;
    }

    // Create optimal tiled target image
    VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
    imageCreateInfo.imageType = VK_IMAGE_TYPE_3D;
    imageCreateInfo.format = texture_.format;
    imageCreateInfo.mipLevels = texture_.mipLevels;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.extent.width = texture_.width;
    imageCreateInfo.extent.height = texture_.height;
    imageCreateInfo.extent.depth = texture_.depth;
    // Set initial layout of the image to undefined
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.usage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VK_CHECK_RESULT(
        vkCreateImage(device_, &imageCreateInfo, nullptr, &texture_.image));

    // Device local memory to back up image
    VkMemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs = {};
    vkGetImageMemoryRequirements(device_, texture_.image, &memReqs);
    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex = vulkanDevice_->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device_, &memAllocInfo, nullptr,
                                     &texture_.deviceMemory));
    VK_CHECK_RESULT(
        vkBindImageMemory(device_, texture_.image, texture_.deviceMemory, 0));

    // Create sampler
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
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(
        vkCreateSampler(device_, &sampler, nullptr, &texture_.sampler));

    // Create image view
    VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
    view.image = texture_.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_3D;
    view.format = texture_.format;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.baseMipLevel = 0;
    view.subresourceRange.baseArrayLayer = 0;
    view.subresourceRange.layerCount = 1;
    view.subresourceRange.levelCount = 1;
    VK_CHECK_RESULT(vkCreateImageView(device_, &view, nullptr, &texture_.view));

    // Fill image descriptor image info to be used descriptor set setup
    texture_.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    texture_.descriptor.imageView = texture_.view;
    texture_.descriptor.sampler = texture_.sampler;

    updateNoiseTexture();
  }

  // Generate randomized noise and upload it to the 3D texture using staging
  void updateNoiseTexture() {
    const uint32_t texMemSize =
        texture_.width * texture_.height * texture_.depth;

    uint8_t* data = new uint8_t[texMemSize];
    memset(data, 0, texMemSize);

    // Generate perlin based noise
    std::cout << "Generating " << texture_.width << " x " << texture_.height
              << " x " << texture_.depth << " noise texture..." << std::endl;

    auto tStart = std::chrono::high_resolution_clock::now();

    PerlinNoise<float> perlinNoise(!benchmark.active);
    FractalNoise<float> fractalNoise(perlinNoise);

    const float noiseScale = static_cast<float>(rand() % 10) + 4.0f;

#pragma omp parallel for
    for (int32_t z = 0; z < static_cast<int32_t>(texture_.depth); z++) {
      for (int32_t y = 0; y < static_cast<int32_t>(texture_.height); y++) {
        for (int32_t x = 0; x < static_cast<int32_t>(texture_.width); x++) {
          float nx = (float)x / (float)texture_.width;
          float ny = (float)y / (float)texture_.height;
          float nz = (float)z / (float)texture_.depth;
          float n = fractalNoise.noise(nx * noiseScale, ny * noiseScale,
                                       nz * noiseScale);
          n = n - floor(n);
          data[x + y * texture_.width + z * texture_.width * texture_.height] =
              static_cast<uint8_t>(floor(n * 255));
        }
      }
    }

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tDiff =
        std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    std::cout << "Done in " << tDiff << "ms" << std::endl;

    // Create a host-visible staging buffer that contains the raw image data
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    // Buffer object
    VkBufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo();
    bufferCreateInfo.size = texMemSize;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK_RESULT(
        vkCreateBuffer(device_, &bufferCreateInfo, nullptr, &stagingBuffer));

    // Allocate host visible memory for data upload
    VkMemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs = {};
    vkGetBufferMemoryRequirements(device_, stagingBuffer, &memReqs);
    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex = vulkanDevice_->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK_RESULT(
        vkAllocateMemory(device_, &memAllocInfo, nullptr, &stagingMemory));
    VK_CHECK_RESULT(
        vkBindBufferMemory(device_, stagingBuffer, stagingMemory, 0));

    // Copy texture data into staging buffer
    uint8_t* mapped;
    VK_CHECK_RESULT(vkMapMemory(device_, stagingMemory, 0, memReqs.size, 0,
                                (void**)&mapped));
    memcpy(mapped, data, texMemSize);
    vkUnmapMemory(device_, stagingMemory);

    VkCommandBuffer copyCmd = vulkanDevice_->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    // The sub resource range describes the regions of the image we will be
    // transitioned
    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.layerCount = 1;

    // Optimal image will be used as destination for the copy, so we must
    // transfer from our initial undefined image layout to the transfer
    // destination layout
    vks::tools::setImageLayout(
        copyCmd, texture_.image, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);

    // Copy 3D noise data to texture

    // Setup buffer copy regions
    VkBufferImageCopy bufferCopyRegion{};
    bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bufferCopyRegion.imageSubresource.mipLevel = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount = 1;
    bufferCopyRegion.imageExtent.width = texture_.width;
    bufferCopyRegion.imageExtent.height = texture_.height;
    bufferCopyRegion.imageExtent.depth = texture_.depth;

    vkCmdCopyBufferToImage(copyCmd, stagingBuffer, texture_.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &bufferCopyRegion);

    // Change texture image layout to shader read after all mip levels have been
    // copied
    texture_.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vks::tools::setImageLayout(copyCmd, texture_.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               texture_.imageLayout, subresourceRange);

    vulkanDevice_->flushCommandBuffer(copyCmd, queue_, true);

    // Clean up staging resources
    delete[] data;
    vkFreeMemory(device_, stagingMemory, nullptr);
    vkDestroyBuffer(device_, stagingBuffer, nullptr);
  }

  // Free all Vulkan resources used a texture object
  void destroyTextureImage(Texture texture) {
    if (texture.view != VK_NULL_HANDLE)
      vkDestroyImageView(device_, texture.view, nullptr);
    if (texture.image != VK_NULL_HANDLE)
      vkDestroyImage(device_, texture.image, nullptr);
    if (texture.sampler != VK_NULL_HANDLE)
      vkDestroySampler(device_, texture.sampler, nullptr);
    if (texture.deviceMemory != VK_NULL_HANDLE)
      vkFreeMemory(device_, texture.deviceMemory, nullptr);
  }

  // Creates a vertex and index buffer for a quad made of two triangles
  // This is used to display the texture on
  void generateQuad() {
    // Setup vertices for a single uv-mapped quad made from two triangles
    std::vector<Vertex> vertices = {
        {{1.0f, 1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
        {{-1.0f, 1.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
        {{-1.0f, -1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{1.0f, -1.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}};

    // Setup indices
    std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};
    indexCount = static_cast<uint32_t>(indices.size());

    // Create buffers and upload data to the GPU
    struct StagingBuffers {
      vks::Buffer vertices;
      vks::Buffer indices;
    } stagingBuffers;

    // Host visible source buffers (staging)
    VK_CHECK_RESULT(vulkanDevice_->createBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &stagingBuffers.vertices, vertices.size() * sizeof(Vertex),
        vertices.data()));
    VK_CHECK_RESULT(vulkanDevice_->createBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &stagingBuffers.indices, indices.size() * sizeof(uint32_t),
        indices.data()));

    // Device local destination buffers
    VK_CHECK_RESULT(vulkanDevice_->createBuffer(
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &vertexBuffer,
        vertices.size() * sizeof(Vertex)));
    VK_CHECK_RESULT(vulkanDevice_->createBuffer(
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &indexBuffer,
        indices.size() * sizeof(uint32_t)));

    // Copy from host do device
    vulkanDevice_->copyBuffer(&stagingBuffers.vertices, &vertexBuffer, queue_);
    vulkanDevice_->copyBuffer(&stagingBuffers.indices, &indexBuffer, queue_);

    // Clean up
    stagingBuffers.vertices.destroy();
    stagingBuffers.indices.destroy();
  }

  void setupDescriptors() {
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              MAX_CONCURRENT_FRAMES),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_CONCURRENT_FRAMES)};
    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(poolSizes,
                                                    MAX_CONCURRENT_FRAMES);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo,
                                           nullptr, &descriptorPool_));

    // Layout
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0 : Vertex shader uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
        // Binding 1 : Fragment shader image sampler
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, 1)};
    VkDescriptorSetLayoutCreateInfo descriptorLayout =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorLayout,
                                                nullptr, &descriptorSetLayout));

    // Image descriptor for the 3D texture
    VkDescriptorImageInfo textureDescriptor =
        vks::initializers::descriptorImageInfo(texture_.sampler, texture_.view,
                                               texture_.imageLayout);

    // Sets per frame, just like the buffers themselves
    // Images do not need to be duplicated per frame, we reuse the same one for
    // each frame
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
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
              &textureDescriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  void preparePipelines() {
    // Layout
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
        vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo,
                                           nullptr, &pipelineLayout));

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

    // Shaders
    shaderStages[0] =
        loadShader(getShadersPath() + "texture3d/texture3d.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] =
        loadShader(getShadersPath() + "texture3d/texture3d.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);

    // Vertex input state
    std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
        vks::initializers::vertexInputBindingDescription(
            0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX)};
    std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
        vks::initializers::vertexInputAttributeDescription(
            0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)),
        vks::initializers::vertexInputAttributeDescription(
            0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)),
        vks::initializers::vertexInputAttributeDescription(
            0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)),
    };
    VkPipelineVertexInputStateCreateInfo vertexInputState =
        vks::initializers::pipelineVertexInputStateCreateInfo();
    vertexInputState.vertexBindingDescriptionCount =
        static_cast<uint32_t>(vertexInputBindings.size());
    vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
    vertexInputState.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(vertexInputAttributes.size());
    vertexInputState.pVertexAttributeDescriptions =
        vertexInputAttributes.data();

    VkGraphicsPipelineCreateInfo pipelineCreateInfo =
        vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass_, 0);
    pipelineCreateInfo.pVertexInputState = &vertexInputState;
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pColorBlendState = &colorBlendState;
    pipelineCreateInfo.pMultisampleState = &multisampleState;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pDepthStencilState = &depthStencilState;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCreateInfo.pStages = shaderStages.data();
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device_, pipelineCache_, 1, &pipelineCreateInfo, nullptr, &pipeline));
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
    uniformData_.projection = camera_.matrices_.perspective;
    uniformData_.modelView = camera_.matrices_.view;
    uniformData_.viewPos = camera_.viewPos_;
    if (!paused) {
      // Animate depth
      uniformData_.depth += frameTimer * 0.15f;
      if (uniformData_.depth > 1.0f) {
        uniformData_.depth = uniformData_.depth - 1.0f;
      }
    }
    memcpy(uniformBuffers_[currentBuffer_].mapped, &uniformData_,
           sizeof(UniformData));
  }

  void prepare() {
    VulkanExampleBase::prepare();
    generateQuad();
    prepareUniformBuffers();
    prepareNoiseTexture(128, 128, 128);
    setupDescriptors();
    preparePipelines();
    prepared_ = true;
  }

  void buildCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();

    VkClearValue clearValues[2]{};
    clearValues[0].color = defaultClearColor;
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

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 0, 1,
                            &descriptorSets_[currentBuffer_], 0, nullptr);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &vertexBuffer.buffer, offsets);
    vkCmdBindIndexBuffer(cmdBuffer, indexBuffer.buffer, 0,
                         VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmdBuffer, indexCount, 1, 0, 0, 0);

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
      if (overlay->button("Generate new texture")) {
        updateNoiseTexture();
      }
    }
  }
};

VULKAN_EXAMPLE_MAIN()
