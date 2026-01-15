/*
 * Vulkan Example - Smoke simulation with compute shaders and cube marching
 *
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#include "VulkanglTFModel.h"
#include "vulkanexamplebase.h"

// Vertex layout for this example
struct Vertex {
  float pos[3];
  float color[3];
};

class VulkanExample : public VulkanExampleBase {
 public:
  VkPhysicalDeviceVulkan13Features enabledFeatures13_{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};

  // Handles rendering the compute outputs
  struct Graphics {
    // families differ and require additional barriers
    uint32_t queueFamilyIndex{0};

    vks::Buffer cubeVerticesBuffer;
    vks::Buffer cubeIndicesBuffer;
    uint32_t indexCount{0};

    struct MarchUBO {
      alignas(16) glm::mat4 model;
      alignas(16) glm::mat4 invModel;
      alignas(16) glm::mat4 cameraView;
      alignas(16) glm::mat4 perspective;
      alignas(16) glm::vec3 cameraPos;
      alignas(8) glm::vec2 screenRes;
      float time{0};
    };

    struct UBO {
      MarchUBO march;
    } ubos_;

    // vk Buffers
    struct UniformBuffers {
      vks::Buffer march;
    };
    std::array<UniformBuffers, MAX_CONCURRENT_FRAMES> uniformBuffers_;

    // Pipelines
    struct {
      VkPipeline rayMarch{VK_NULL_HANDLE};
    } pipelines_{};
    struct {
      VkPipelineLayout rayMarch{VK_NULL_HANDLE};
    } pipelineLayouts_;

    // Descriptors
    struct {
      VkDescriptorSetLayout rayMarch{VK_NULL_HANDLE};
    } descriptorSetLayouts_;
    struct DescriptorSets {
      VkDescriptorSet rayMarch{VK_NULL_HANDLE};
    };
    std::array<DescriptorSets, MAX_CONCURRENT_FRAMES> descriptorSets_{};

  } graphics_;

  // Handles all compute pipelines
  struct Compute {
#define VECTOR_FIELD_FORMAT VK_FORMAT_R32G32B32A32_SFLOAT
#define SCALAR_FIELD_FORMAT VK_FORMAT_R32_SFLOAT
#define COMPUTE_TEXTURE_DIMENSIONS 128

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

    // uniform buffers
    std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers_;

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

    // 3D textures neeeded to store vector/scalar states
    // Split into read/write respectively
    static const int texture_count = 6;
    // Texture index mappings
    // 0 velocity
    // 1 pressure
    // 2 divergence
    // 3 vorticity
    // 4 density
    // 5 temperature
    std::array<Texture3D, texture_count> read_textures;
    std::array<Texture3D, texture_count> write_textures;

    // Buffers
    std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers;

    // Pipelines
    struct {
      VkPipeline advect;
      VkPipeline divergence;
      VkPipeline jacobi;
      VkPipeline gradient;
      VkPipeline vorticity;
      VkPipeline vorticityConfinement;
    } pipelines_{};
    struct {
      VkPipelineLayout advect{VK_NULL_HANDLE};
      VkPipelineLayout divergence{VK_NULL_HANDLE};
      VkPipelineLayout jacobi{VK_NULL_HANDLE};
      VkPipelineLayout gradient{VK_NULL_HANDLE};
      VkPipelineLayout vorticity{VK_NULL_HANDLE};
      VkPipelineLayout vorticityConfintement{VK_NULL_HANDLE};
    } pipelineLayouts_;

    // Descriptors
    struct {
      VkDescriptorSetLayout advect{VK_NULL_HANDLE};
      VkDescriptorSetLayout divergence{VK_NULL_HANDLE};
      VkDescriptorSetLayout jacobi{VK_NULL_HANDLE};
      VkDescriptorSetLayout gradient{VK_NULL_HANDLE};
      VkDescriptorSetLayout vorticity{VK_NULL_HANDLE};
      VkDescriptorSetLayout vorticityConfinement{VK_NULL_HANDLE};
    } descriptorSetLayouts_;
    struct DescriptorSets {
      VkDescriptorSet advect{VK_NULL_HANDLE};
      VkDescriptorSet divergence{VK_NULL_HANDLE};
      VkDescriptorSet jacobi{VK_NULL_HANDLE};
      VkDescriptorSet gradient{VK_NULL_HANDLE};
      VkDescriptorSet vorticity{VK_NULL_HANDLE};
      VkDescriptorSet vorticityConfinement{VK_NULL_HANDLE};
    };
    std::array<DescriptorSets, MAX_CONCURRENT_FRAMES> descriptorSets_{};

  } compute_;

  void prepareComputeTexture(Compute::Texture3D& texture,
                             bool readOnly,
                             VkFormat texture_format) {
    // A 3D texture is described as width x height x depth
    texture.width = COMPUTE_TEXTURE_DIMENSIONS;
    texture.height = COMPUTE_TEXTURE_DIMENSIONS;
    texture.depth = COMPUTE_TEXTURE_DIMENSIONS;
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
        (readOnly)
            ? (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            : (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

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

    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0 : Array of INPUT textures to advect
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_COMPUTE_BIT, /*binding id*/ 0,
            (uint32_t)compute_.read_textures.size()),
        // Binding 1 : Array of OUTPUT textures to write result
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 1, (uint32_t)compute_.write_textures.size()),
        // Binding 2 : Single velocity field texture
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_COMPUTE_BIT,
            /*binding id*/ 2, 1)};
    VkDescriptorSetLayoutCreateInfo descriptorLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);

    // Flags for descriptor arrays - typically want partially bound
    std::vector<VkDescriptorBindingFlags> bindingFlags = {
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,

        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT};

    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bindingFlagsInfo.pNext = nullptr;
    bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
    bindingFlagsInfo.pBindingFlags = bindingFlags.data();

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
                                    &compute_.descriptorSetLayouts_.advect));
  }

  void prepareCompute() {
    // Create a compute capable device queue
    vkGetDeviceQueue(device_, compute_.queueFamilyIndex, 0, &compute_.queue);

    createTexturesForDescriptorIndexing();

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

    for (auto i = 0; i < compute_.uniformBuffers.size(); i++) {
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool_, &compute_.descriptorSetLayouts_.advect, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &compute_.descriptorSets_[i].advect));

      readOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].advect;
      writeOnlyTextureArrayDescriptor.dstSet =
          compute_.descriptorSets_[i].advect;

      std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
          readOnlyTextureArrayDescriptor,
          writeOnlyTextureArrayDescriptor,
          vks::initializers::writeDescriptorSet(
              compute_.descriptorSets_[i].advect,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, /*binding id*/ 2,
              &compute_.read_textures[0].descriptor),
      };
      vkUpdateDescriptorSets(
          device_, static_cast<uint32_t>(computeWriteDescriptorSets.size()),
          computeWriteDescriptorSets.data(), 0, nullptr);
    }

    // Create pipelines
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &compute_.descriptorSetLayouts_.advect, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo,
                                           nullptr,
                                           &compute_.pipelineLayouts_.advect));
    VkComputePipelineCreateInfo computePipelineCreateInfo =
        vks::initializers::computePipelineCreateInfo(
            compute_.pipelineLayouts_.advect, 0);

    // 1st pass
    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "smoke/advect.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);

    // We want to use as much shared memory for the compute shader invocations
    // as available, so we calculate it based on the device limits and pass it
    // to the shader via specialization constants
    uint32_t sharedDataSize = std::min(
        (uint32_t)1024,
        (uint32_t)(vulkanDevice_->properties.limits.maxComputeSharedMemorySize /
                   sizeof(glm::vec4)));
    VkSpecializationMapEntry specializationMapEntry =
        vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
    VkSpecializationInfo specializationInfo =
        vks::initializers::specializationInfo(1, &specializationMapEntry,
                                              sizeof(int32_t), &sharedDataSize);
    computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;

    VK_CHECK_RESULT(vkCreateComputePipelines(
        device_, pipelineCache_, 1, &computePipelineCreateInfo, nullptr,
        &compute_.pipelines_.advect));

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

  void setupDescriptors() {
    // Layout: Ray march
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0 : Vertex shader uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding_id*/ 0),
        // Binding 1 : Volume texture to render
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding_id*/ 1),
    };

    VkDescriptorSetLayoutCreateInfo descriptorLayout =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorLayout, nullptr,
                                    &graphics_.descriptorSetLayouts_.rayMarch));

    // Sets per frame, just like the buffers themselves
    // Images do not need to be duplicated per frame, we reuse the same one
    // for each frame
    for (auto i = 0; i < graphics_.uniformBuffers_.size(); i++) {
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool_, &graphics_.descriptorSetLayouts_.rayMarch, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &graphics_.descriptorSets_[i].rayMarch));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          // Binding 0 : Projection/View matrix as uniform buffer
          vks::initializers::writeDescriptorSet(
              graphics_.descriptorSets_[i].rayMarch,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &graphics_.uniformBuffers_[i].march.descriptor),
          // Binding 1 : Volumetric texture to visualize
          vks::initializers::writeDescriptorSet(
              graphics_.descriptorSets_[i].rayMarch,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &compute_.read_textures[0].descriptor),
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
            offsetof(Vertex, color)),  // Location 1 : Color
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

  void prepareDescriptorPool() {
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            /*total ubo count */ /*graphics*/ 1 * MAX_CONCURRENT_FRAMES),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            /*total texture count (across all pipelines) */ (
                /*graphics: volume texture*/ 1 +
                /*compute textures*/ 1 + compute_.read_textures.size()) *
                MAX_CONCURRENT_FRAMES),
        // textures for writing
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            compute_.write_textures.size() * MAX_CONCURRENT_FRAMES)};

    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(
            poolSizes,
            /*total descriptor count*/ (/*graphics*/ 1 + /*compute*/ 1) *
                MAX_CONCURRENT_FRAMES);
    // Needed if using VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT in descriptor
    // bindings
    descriptorPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo,
                                           nullptr, &descriptorPool_));
  }

  // Prepare and initialize uniform buffer containing shader uniforms
  void prepareUniformBuffers() {
    for (auto& buffer : graphics_.uniformBuffers_) {
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.march, sizeof(Graphics::MarchUBO), &graphics_.ubos_.march));
      VK_CHECK_RESULT(buffer.march.map());
    }
  }

  void updateUniformBuffers() {
    graphics_.ubos_.march.model = glm::mat4(1.0);
    graphics_.ubos_.march.invModel = glm::inverse(glm::mat4(1.0));
    graphics_.ubos_.march.cameraView = camera_.matrices_.view;
    graphics_.ubos_.march.screenRes = glm::vec2(width_, height_);
    graphics_.ubos_.march.perspective = camera_.matrices_.perspective;
    graphics_.ubos_.march.cameraPos = camera_.position_;
    graphics_.ubos_.march.time =
        std::chrono::duration<float>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    memcpy(graphics_.uniformBuffers_[currentBuffer_].march.mapped,
           &graphics_.ubos_.march, sizeof(Graphics::MarchUBO));
  }

  void prepareGraphics() {
    generateCube();
    prepareUniformBuffers();
    setupDescriptors();
    preparePipelines();
  }

  void prepare() {
    VulkanExampleBase::prepare();
    graphics_.queueFamilyIndex = vulkanDevice_->queueFamilyIndices.graphics;
    compute_.queueFamilyIndex = vulkanDevice_->queueFamilyIndices.compute;
    prepareDescriptorPool();
    prepareCompute();
    prepareGraphics();
    prepared_ = true;
  }

  virtual void render() {
    if (!prepared_) {
      return;
    }
    // Use a fence to ensure that compute command buffer has finished executing
    // before using it again
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

    VulkanExampleBase::prepareFrame();
    updateUniformBuffers();
    buildGraphicsCommandBuffer();
    VulkanExampleBase::submitFrame();
  }

  void buildComputeCommandBuffer() {
    VkCommandBuffer cmdBuffer = compute_.commandBuffers[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();

    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    // Layout transition for both read and write texture maps
    for (int i = 0; i < compute_.texture_count; i++) {
      // Read textures
      vks::tools::insertImageMemoryBarrier(
          cmdBuffer, compute_.read_textures[i].image, 0,
          VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
          VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

      // Write textures
      vks::tools::insertImageMemoryBarrier(
          cmdBuffer, compute_.write_textures[i].image, 0,
          VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
          VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    }
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute_.pipelines_.advect);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_.pipelineLayouts_.advect, 0, 1,
                            &compute_.descriptorSets_[currentBuffer_].advect, 0,
                            nullptr);

    vkCmdDispatch(cmdBuffer, compute_.write_textures[0].width / 8,
                  compute_.write_textures[0].height / 8,
                  compute_.write_textures[0].depth / 8);

    for (int i = 0; i < compute_.texture_count; i++) {
      swapTextures(cmdBuffer, compute_.write_textures[i].image,
                   compute_.read_textures[i].image);
    }

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void swapTextures(VkCommandBuffer& cmdBuffer,
                    VkImage& srcImage,
                    VkImage& dstImage) {
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

    copyRegion.extent.width = COMPUTE_TEXTURE_DIMENSIONS;
    copyRegion.extent.height = COMPUTE_TEXTURE_DIMENSIONS;
    copyRegion.extent.depth = COMPUTE_TEXTURE_DIMENSIONS;

    // Copy output of write to read buffer
    vkCmdCopyImage(cmdBuffer, srcImage, VK_IMAGE_LAYOUT_GENERAL, dstImage,
                   VK_IMAGE_LAYOUT_GENERAL, 1, &copyRegion);
  }

  void buildGraphicsCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    // preMarchCmdFront(cmdBuffer);
    // preMarchCmdBack(cmdBuffer);
    rayMarchCmd(cmdBuffer);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void rayMarchCmd(VkCommandBuffer& cmdBuffer) {
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
    vkCmdSetCullMode(cmdBuffer, VkCullModeFlagBits(VK_CULL_MODE_NONE));

    VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1,
                           &graphics_.cubeVerticesBuffer.buffer, offsets);
    vkCmdBindIndexBuffer(cmdBuffer, graphics_.cubeIndicesBuffer.buffer, 0,
                         VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmdBuffer, graphics_.indexCount, 1, 0, 0, 0);
    drawUI(cmdBuffer);

    // End dynamic rendering
    vkCmdEndRendering(cmdBuffer);
  }

  void generateCube() {
    // Setup vertices indices for a colored cube
    std::vector<Vertex> vertices = {
        {{-0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}},
        {{0.5f, -0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
        {{0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        {{-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 0.0f}},
        {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
        {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
        {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}},
        {{-0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}},
    };

    std::vector<uint32_t> indices = {
        0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7,
        4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3,
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

  virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay) {}

  void setupRenderPass() override {
    // With VK_KHR_dynamic_rendering we no longer need a render pass, so skip
    // the sample base render pass setup
    renderPass_ = VK_NULL_HANDLE;
  }

  void setupFrameBuffer() override {
    // With VK_KHR_dynamic_rendering we no longer need a frame buffer, so skip
    // the sample base framebuffer setup
  }

  void getEnabledFeatures() override {}

  void getEnabledExtensions() override {}

  VulkanExample() : VulkanExampleBase() {
    title_ = "Smoke Simulation";
    camera_.type_ = Camera::CameraType::lookat;
    camera_.setMovementSpeed(25.f);
    camera_.setPosition(glm::vec3(0.0f, 0.0f, -2.f));
    camera_.setRotation(glm::vec3(0.0f, 15.0f, 0.0f));
    camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);

    apiVersion_ = VK_API_VERSION_1_3;
    enabledFeatures13_.dynamicRendering = VK_TRUE;
    deviceCreatepNextChain_ = &enabledFeatures13_;
  }

  ~VulkanExample() {
    if (device_) {
      // Graphics
      vkDestroyPipeline(device_, graphics_.pipelines_.rayMarch, nullptr);
      vkDestroyPipelineLayout(device_, graphics_.pipelineLayouts_.rayMarch,
                              nullptr);
      vkDestroyDescriptorSetLayout(
          device_, graphics_.descriptorSetLayouts_.rayMarch, nullptr);

      for (auto& buffer : graphics_.uniformBuffers_) {
        buffer.march.destroy();
      }
      graphics_.cubeVerticesBuffer.destroy();
      graphics_.cubeIndicesBuffer.destroy();

      // Compute
      // Textures
      for (auto& texture : compute_.read_textures) {
        if (texture.view != VK_NULL_HANDLE)
          vkDestroyImageView(device_, texture.view, nullptr);
        if (texture.image != VK_NULL_HANDLE)
          vkDestroyImage(device_, texture.image, nullptr);
        if (texture.sampler != VK_NULL_HANDLE)
          vkDestroySampler(device_, texture.sampler, nullptr);
        if (texture.deviceMemory != VK_NULL_HANDLE)
          vkFreeMemory(device_, texture.deviceMemory, nullptr);
      }
      for (auto& texture : compute_.write_textures) {
        if (texture.view != VK_NULL_HANDLE)
          vkDestroyImageView(device_, texture.view, nullptr);
        if (texture.image != VK_NULL_HANDLE)
          vkDestroyImage(device_, texture.image, nullptr);
        if (texture.sampler != VK_NULL_HANDLE)
          vkDestroySampler(device_, texture.sampler, nullptr);
        if (texture.deviceMemory != VK_NULL_HANDLE)
          vkFreeMemory(device_, texture.deviceMemory, nullptr);
      }

      vkDestroyPipeline(device_, compute_.pipelines_.advect, nullptr);
      vkDestroyPipelineLayout(device_, compute_.pipelineLayouts_.advect,
                              nullptr);
      vkDestroyDescriptorSetLayout(
          device_, compute_.descriptorSetLayouts_.advect, nullptr);
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
