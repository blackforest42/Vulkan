/*
 * Vulkan Example - Cube map texture loading and displaying
 *
 * This sample shows how to load and render a cubemap. A cubemap is a textures
 * that contains 6 images, one per cube face. The sample displays the cubemap as
 * a skybox (background) and as a reflection on a selectable object
 *
 * Copyright (C) 2016-2025 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#include "vulkanexamplebase.h"

#include <ktx.h>
#include <ktxvulkan.h>
#include "VulkanglTFModel.h"
#include "stb_image.h"

// Offscreen frame buffer properties
// #define FB_COLOR_FORMAT VK_FORMAT_R8G8B8A8_UNORM
#define FB_COLOR_FORMAT VK_FORMAT_R16G16B16A16_SFLOAT
// Number of down/up samples during bloom
constexpr int NUM_SAMPLE_SIZES = 8;

class VulkanExample : public VulkanExampleBase {
 public:
  vks::TextureCubeMap cubeMap_{};
  vks::Texture accretionDiskTextureMap_{};

  // Hacky workaround.
  // Boolean uniforms don't work in shaders for some reason.
  bool showBlackholeUI = true;
  bool gravatationalLensingEnabled = true;
  bool accDiskEnabled = true;
  bool accDiskParticleEnabled = true;
  bool toneMappingEnabled = true;
  bool karisAverageEnabled = true;

  struct BlackholeUBO {
    alignas(16) glm::mat4 cameraView;
    alignas(16) glm::vec3 cameraPos;
    alignas(8) glm::vec2 resolution;

    // Epoch time in seconds
    float time;

    int showBlackhole;
    int gravatationalLensingEnabled;
    int accDiskEnabled;
    int accDiskParticleEnabled;

    float accDiskHeight{0.55f};
    float accDiskLit{0.25f};
    float accDiskDensityV{2.0f};
    float accDiskDensityH{4.0f};
    float accDiskNoiseScale{.8f};
    int accDiskNoiseLOD{5};
    float accDiskSpeed{0.5f};
  };

  struct BrightnessUBO {
    glm::vec2 viewportResolution;
  };

  struct DownsampleUBO {
    alignas(8) glm::vec2 srcResolution;
  };

  struct UpsampleUBO {
    alignas(8) glm::vec2 dstResolution;
  };

  struct BlendUBO {
    // Tonemapping
    alignas(4) int tonemappingEnabled;
    alignas(4) float exposure{1.0f};

    // Bloom
    // Linear interpolation b/w source tex and bloom tex
    alignas(4) float bloomStrength{0.1f};
  };

  struct {
    BlackholeUBO blackhole;
    BrightnessUBO brightness;
    DownsampleUBO downsample;
    UpsampleUBO upsample;
    BlendUBO blend;
  } ubos_;

  struct UniformBuffers {
    vks::Buffer blackhole;
    vks::Buffer brightness;
    std::array<vks::Buffer, NUM_SAMPLE_SIZES> downsample;
    std::array<vks::Buffer, NUM_SAMPLE_SIZES> upsample;
    vks::Buffer blend;
  };
  std::array<UniformBuffers, MAX_CONCURRENT_FRAMES> uniformBuffers_{};

  struct {
    VkPipelineLayout blackhole;
    VkPipelineLayout brightness;
    VkPipelineLayout downsample;
    VkPipelineLayout upsample;
    VkPipelineLayout blend;
  } pipelineLayouts_{};

  struct {
    VkPipeline blackhole;
    VkPipeline brightness;
    VkPipeline downsample;
    VkPipeline upsample;
    VkPipeline blend;
  } pipelines_{};

  struct {
    VkDescriptorSetLayout blackhole;
    VkDescriptorSetLayout brightness;
    VkDescriptorSetLayout downsample;
    VkDescriptorSetLayout upsample;
    VkDescriptorSetLayout blend;
  } descriptorSetLayouts_{};

  struct DescriptorSets {
    VkDescriptorSet blackhole;
    VkDescriptorSet brightness;
    std::array<VkDescriptorSet, NUM_SAMPLE_SIZES> downsamples{};
    std::array<VkDescriptorSet, NUM_SAMPLE_SIZES> upsamples{};
    VkDescriptorSet blend;
  };
  std::array<DescriptorSets, MAX_CONCURRENT_FRAMES> descriptorSets_{};

  // Framebuffer for offscreen rendering
  struct FrameBufferAttachment {
    VkImage image;
    VkDeviceMemory mem;
    VkImageView view;
  };
  struct FrameBuffer {
    int32_t width, height;
    VkFramebuffer framebuffer;
    FrameBufferAttachment color;
    VkDescriptorImageInfo descriptor;
  };
  struct OffscreenPass {
    VkRenderPass renderPass;
    VkSampler sampler;
    // Holds the first offscreen framebuffer
    // used by first down sample pass
    FrameBuffer original;
    FrameBuffer brightness;
    // Holds all downsampled framebuffers
    std::array<FrameBuffer, NUM_SAMPLE_SIZES> down_samples;
    std::array<FrameBuffer, NUM_SAMPLE_SIZES> up_samples;
    // Holds final bloom framebuffer
    // used by last up sample pass
    FrameBuffer final;
  } offscreenPass_{};

  // (A.2)
  // Prepare and initialize uniform buffer containing shader uniforms
  void prepareUniformBuffers() {
    for (auto& buffer : uniformBuffers_) {
      // Blackhole
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.blackhole, sizeof(BlackholeUBO), &ubos_.blackhole));
      VK_CHECK_RESULT(buffer.blackhole.map());

      // Brightness
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.brightness, sizeof(BrightnessUBO), &ubos_.brightness));
      VK_CHECK_RESULT(buffer.brightness.map());

      // Downsample
      for (int i = 0; i < NUM_SAMPLE_SIZES; i++) {
        VK_CHECK_RESULT(vulkanDevice_->createBuffer(
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &buffer.downsample[i], sizeof(DownsampleUBO), &ubos_.downsample));
        VK_CHECK_RESULT(buffer.downsample[i].map());
      }

      // Upsample
      for (int i = 0; i < NUM_SAMPLE_SIZES; i++) {
        VK_CHECK_RESULT(vulkanDevice_->createBuffer(
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &buffer.upsample[i], sizeof(UpsampleUBO), &ubos_.upsample));
        VK_CHECK_RESULT(buffer.upsample[i].map());
      }

      // Blend
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.blend, sizeof(BlendUBO), &ubos_.blend));
      VK_CHECK_RESULT(buffer.blend.map());
    }
  }

  // (A.3) Prepare the offscreen framebuffers used for up/down sampling
  void prepareOffscreen() {
    // Create a separate render pass for the offscreen rendering as it may
    // differ from the one used for scene rendering
    VkAttachmentDescription attchmentDescriptions;
    // Color attachment
    attchmentDescriptions.format = FB_COLOR_FORMAT;
    attchmentDescriptions.samples = VK_SAMPLE_COUNT_1_BIT;
    attchmentDescriptions.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attchmentDescriptions.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attchmentDescriptions.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attchmentDescriptions.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attchmentDescriptions.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attchmentDescriptions.finalLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorReference = {
        0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpassDescription = {};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorReference;

    // Use subpass dependencies for layout transitions
    std::array<VkSubpassDependency, 3> dependencies{};

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask =
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dstAccessMask =
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    dependencies[0].dependencyFlags = 0;

    dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].dstSubpass = 0;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].dstStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[2].srcSubpass = 0;
    dependencies[2].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[2].srcStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[2].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[2].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[2].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // Create the actual renderpass
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &attchmentDescriptions;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpassDescription;
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();
    VK_CHECK_RESULT(vkCreateRenderPass(device_, &renderPassInfo, nullptr,
                                       &offscreenPass_.renderPass));

    // Create sampler to sample from the color attachments
    VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = sampler.addressModeU;
    sampler.addressModeW = sampler.addressModeU;
    sampler.mipLodBias = 0.0f;
    sampler.maxAnisotropy = 1.0f;
    sampler.minLod = 0.0f;
    sampler.maxLod = 1.0f;
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(
        vkCreateSampler(device_, &sampler, nullptr, &offscreenPass_.sampler));

    // Generate framebuffer for first renderpass
    offscreenPass_.original.height = height_;
    offscreenPass_.original.width = width_;
    prepareOffscreenFramebuffer(&offscreenPass_.original, FB_COLOR_FORMAT);

    // Generate framebuffer for brightness pass
    offscreenPass_.brightness.height = height_;
    offscreenPass_.brightness.width = width_;
    prepareOffscreenFramebuffer(&offscreenPass_.brightness, FB_COLOR_FORMAT);

    // Generate fb's for each down sample using mipmap scaling (1/2).
    for (int i = 0; i < NUM_SAMPLE_SIZES; i++) {
      offscreenPass_.down_samples[i].height =
          static_cast<uint32_t>(height_ >> (i + 1));
      offscreenPass_.down_samples[i].width =
          static_cast<uint32_t>(width_ >> (i + 1));
      prepareOffscreenFramebuffer(&offscreenPass_.down_samples[i],
                                  FB_COLOR_FORMAT);

      offscreenPass_.up_samples[i].height = static_cast<uint32_t>(height_ >> i);
      offscreenPass_.up_samples[i].width = static_cast<uint32_t>(width_ >> i);
      prepareOffscreenFramebuffer(&offscreenPass_.up_samples[i],
                                  FB_COLOR_FORMAT);
    }

    // Generate framebuffer for last renderpass
    offscreenPass_.final.height = height_;
    offscreenPass_.final.width = width_;
    prepareOffscreenFramebuffer(&offscreenPass_.final, FB_COLOR_FORMAT);
  }

  void prepareOffscreenFramebuffer(FrameBuffer* frameBuf,
                                   VkFormat colorFormat) {
    // Color attachment
    VkImageCreateInfo imageCI = vks::initializers::imageCreateInfo();
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = colorFormat;
    imageCI.extent.width = frameBuf->width;
    imageCI.extent.height = frameBuf->height;
    imageCI.extent.depth = 1;
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    // We will sample directly from the color attachment
    imageCI.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs;

    VkImageViewCreateInfo colorImageView =
        vks::initializers::imageViewCreateInfo();
    colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
    colorImageView.format = colorFormat;
    colorImageView.flags = 0;
    colorImageView.subresourceRange = {};
    colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorImageView.subresourceRange.baseMipLevel = 0;
    colorImageView.subresourceRange.levelCount = 1;
    colorImageView.subresourceRange.baseArrayLayer = 0;
    colorImageView.subresourceRange.layerCount = 1;

    VK_CHECK_RESULT(
        vkCreateImage(device_, &imageCI, nullptr, &frameBuf->color.image));
    vkGetImageMemoryRequirements(device_, frameBuf->color.image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = vulkanDevice_->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(
        vkAllocateMemory(device_, &memAlloc, nullptr, &frameBuf->color.mem));
    VK_CHECK_RESULT(vkBindImageMemory(device_, frameBuf->color.image,
                                      frameBuf->color.mem, 0));

    colorImageView.image = frameBuf->color.image;
    VK_CHECK_RESULT(vkCreateImageView(device_, &colorImageView, nullptr,
                                      &frameBuf->color.view));

    VkImageView attachments[1]{frameBuf->color.view};

    VkFramebufferCreateInfo fbufCreateInfo =
        vks::initializers::framebufferCreateInfo();
    fbufCreateInfo.renderPass = offscreenPass_.renderPass;
    fbufCreateInfo.attachmentCount = 1;
    fbufCreateInfo.pAttachments = attachments;
    fbufCreateInfo.width = frameBuf->width;
    fbufCreateInfo.height = frameBuf->height;
    fbufCreateInfo.layers = 1;

    VK_CHECK_RESULT(vkCreateFramebuffer(device_, &fbufCreateInfo, nullptr,
                                        &frameBuf->framebuffer));

    // Fill a descriptor for later use in a descriptor set
    frameBuf->descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    frameBuf->descriptor.imageView = frameBuf->color.view;
    frameBuf->descriptor.sampler = offscreenPass_.sampler;
  }

  // (A.4)
  void setupDescriptors() {
    // Pool
    // BUG: Magic numbers 8, 6, 4, up ahead. Removing them causes runtime error.
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            /* descriptorCount */ MAX_CONCURRENT_FRAMES * 10 *
                NUM_SAMPLE_SIZES),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            /* descriptorCount */ MAX_CONCURRENT_FRAMES * 8 *
                NUM_SAMPLE_SIZES)};
    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(
            poolSizes, MAX_CONCURRENT_FRAMES * 6 * NUM_SAMPLE_SIZES);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo,
                                           nullptr, &descriptorPool_));

    // Layout: Blackhole
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0 : Fragment shader blackhole uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),

        // Binding 1 : Fragment shader cubemap
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, /*binding id*/ 1),

        // Binding 2 : Fragment shader blackhole 2D texture
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 2)};

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(
            setLayoutBindings.data(),
            static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.blackhole));

    // Layout: Brightness
    setLayoutBindings = {
        // Binding 0 : Fragment shader brightness uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),

        // Binding 1 : Fragment shader blackhole 2D texture
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 1)};

    descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.brightness));

    // Layout: Downsample
    setLayoutBindings = {
        // Binding 0: uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),

        // Binding 1: Source texture to downsample
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, /*binding id*/ 1)};
    descriptorSetLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.downsample));

    // Layout: Upsample
    setLayoutBindings = {
        // Binding 0: uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),

        // Binding 1: Source texture to upsample
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, /*binding id*/ 1, 1),

        // Binding 2: Source texture to upsample
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, /*binding id*/ 2, 1)};
    descriptorSetLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.upsample));
    // Layout: Blend
    setLayoutBindings = {
        // Binding 0 : uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),

        // Binding 1 : up sampled texture map
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, /*binding id*/ 1),

        // Binding 2 : down sample texture map
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, /*binding id*/ 2)};
    descriptorSetLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI,
                                                nullptr,
                                                &descriptorSetLayouts_.blend));

    // Image descriptor for the cube map texture
    VkDescriptorImageInfo textureDescriptor =
        vks::initializers::descriptorImageInfo(cubeMap_.sampler, cubeMap_.view,
                                               cubeMap_.imageLayout);
    // Image descriptor for the blackhole color texture
    VkDescriptorImageInfo accretionDiskTextureMap =
        vks::initializers::descriptorImageInfo(
            accretionDiskTextureMap_.sampler, accretionDiskTextureMap_.view,
            accretionDiskTextureMap_.imageLayout);

    for (auto i = 0; i < uniformBuffers_.size(); i++) {
      // Descriptor for blackhole
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool_, &descriptorSetLayouts_.blackhole, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].blackhole));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].blackhole, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].blackhole.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].blackhole,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, /*binding id*/ 1,
              &textureDescriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].blackhole,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, /*binding id*/ 2,
              &accretionDiskTextureMap),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Descriptor for brightness pass
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.brightness, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].brightness));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].brightness, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].brightness.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].brightness,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, /*binding id*/ 1,
              &offscreenPass_.original.descriptor)};
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Descriptor for downsample
      for (int sample_level = 0; sample_level < NUM_SAMPLE_SIZES;
           sample_level++) {
        allocInfo = vks::initializers::descriptorSetAllocateInfo(
            descriptorPool_, &descriptorSetLayouts_.downsample, 1);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(
            device_, &allocInfo,
            &descriptorSets_[i].downsamples[sample_level]));
        writeDescriptorSets = {
            vks::initializers::writeDescriptorSet(
                descriptorSets_[i].downsamples[sample_level],
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                /*binding id*/ 0,
                &uniformBuffers_[i].downsample[sample_level].descriptor),
            vks::initializers::writeDescriptorSet(
                descriptorSets_[i].downsamples[sample_level],
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, /*binding id*/ 1,
                sample_level == 0
                    ? &offscreenPass_.brightness.descriptor
                    : &offscreenPass_.down_samples[sample_level - 1]
                           .descriptor)};
        vkUpdateDescriptorSets(
            device_, static_cast<uint32_t>(writeDescriptorSets.size()),
            writeDescriptorSets.data(), 0, nullptr);
      }

      // Descriptor for upsample
      for (int sample_level = NUM_SAMPLE_SIZES - 1; sample_level >= 0;
           sample_level--) {
        allocInfo = vks::initializers::descriptorSetAllocateInfo(
            descriptorPool_, &descriptorSetLayouts_.upsample, 1);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(
            device_, &allocInfo, &descriptorSets_[i].upsamples[sample_level]));
        writeDescriptorSets = {
            vks::initializers::writeDescriptorSet(
                descriptorSets_[i].upsamples[sample_level],
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                /*binding id*/ 0,
                &uniformBuffers_[i].upsample[sample_level].descriptor),

            vks::initializers::writeDescriptorSet(
                descriptorSets_[i].upsamples[sample_level],
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, /*binding id*/ 1,
                sample_level == NUM_SAMPLE_SIZES - 1
                    ? &offscreenPass_.down_samples[sample_level].descriptor
                    : &offscreenPass_.up_samples[sample_level + 1].descriptor),

            vks::initializers::writeDescriptorSet(
                descriptorSets_[i].upsamples[sample_level],
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, /*binding id*/ 2,
                sample_level == 0
                    ? &offscreenPass_.brightness.descriptor
                    : &offscreenPass_.down_samples[sample_level - 1]
                           .descriptor),
        };
        vkUpdateDescriptorSets(
            device_, static_cast<uint32_t>(writeDescriptorSets.size()),
            writeDescriptorSets.data(), 0, nullptr);
      }

      // Descriptor for blend
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.blend, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].blend));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].blend, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].blend.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].blend,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &offscreenPass_.original.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].blend,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 2,  // &offscreenPass_.final.descriptor)};
              &offscreenPass_.up_samples[0].descriptor)};
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  // (A.5)
  void preparePipelines() {
    // Layout
    VkPipelineLayoutCreateInfo pipelineLayoutCI =
        vks::initializers::pipelineLayoutCreateInfo(
            &descriptorSetLayouts_.blackhole, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.blackhole));
    pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.brightness, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.brightness));
    pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.downsample, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.downsample));
    pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.upsample, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.upsample));
    pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.blend, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.blend));

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
        vks::initializers::pipelineCreateInfo(pipelineLayouts_.blend,
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

    // Blend pipeline
    // No vertices to input, quad verts are hardcoded in vertex shader.
    VkPipelineVertexInputStateCreateInfo emptyInputState =
        vks::initializers::pipelineVertexInputStateCreateInfo();
    pipelineCI.pVertexInputState = &emptyInputState;
    pipelineCI.layout = pipelineLayouts_.blend;
    shaderStages[0] = loadShader(getShadersPath() + "blackhole/simple.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "blackhole/blend.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.blend));

    // Blackhole pipeline
    pipelineCI.layout = pipelineLayouts_.blackhole;
    pipelineCI.renderPass = offscreenPass_.renderPass;
    shaderStages[1] =
        loadShader(getShadersPath() + "blackhole/blackhole.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.blackhole));
    // Brightness pipeline
    pipelineCI.layout = pipelineLayouts_.brightness;
    pipelineCI.renderPass = offscreenPass_.renderPass;
    shaderStages[1] =
        loadShader(getShadersPath() + "blackhole/brightness.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.brightness));

    // Downsample pipeline
    pipelineCI.layout = pipelineLayouts_.downsample;
    pipelineCI.renderPass = offscreenPass_.renderPass;
    shaderStages[1] =
        loadShader(getShadersPath() + "blackhole/downsample.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.downsample));

    // Upsample pipeline
    pipelineCI.layout = pipelineLayouts_.upsample;
    pipelineCI.renderPass = offscreenPass_.renderPass;
    // Additive blending
    blendAttachmentState.colorWriteMask = 0xF;
    blendAttachmentState.blendEnable = VK_TRUE;
    blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    shaderStages[0] = loadShader(getShadersPath() + "blackhole/simple.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] =
        loadShader(getShadersPath() + "blackhole/upsample.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.upsample));
  }

  // (B) Called in VulkanExampleBase::renderLoop()
  virtual void render() {
    if (!prepared_)
      return;
    VulkanExampleBase::prepareFrame();
    updateUniformBuffers();
    buildCommandBuffer();
    VulkanExampleBase::submitFrame();
  }

  // (B.1)
  void updateUniformBuffers() {
    ubos_.blackhole.cameraView = camera_.matrices_.view;
    ubos_.blackhole.cameraPos = camera_.position_;
    ubos_.blackhole.time =
        std::chrono::duration<float>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    ubos_.blackhole.resolution = glm::vec2(offscreenPass_.original.width,
                                           offscreenPass_.original.height);
    ubos_.blackhole.showBlackhole = showBlackholeUI;
    ubos_.blackhole.gravatationalLensingEnabled = gravatationalLensingEnabled;
    ubos_.blackhole.accDiskEnabled = accDiskEnabled;
    ubos_.blackhole.accDiskParticleEnabled = accDiskParticleEnabled;
    memcpy(uniformBuffers_[currentBuffer_].blackhole.mapped, &ubos_.blackhole,
           sizeof(BlackholeUBO));

    ubos_.brightness.viewportResolution = glm::vec2(
        offscreenPass_.brightness.width, offscreenPass_.brightness.height);
    memcpy(uniformBuffers_[currentBuffer_].brightness.mapped, &ubos_.brightness,
           sizeof(BrightnessUBO));

    ubos_.blend.tonemappingEnabled = toneMappingEnabled;
    memcpy(uniformBuffers_[currentBuffer_].blend.mapped, &ubos_.blend,
           sizeof(BlendUBO));
  }

  // (B.2)
  void buildCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    // Blackhole
    {
      VkClearValue clearValues{};
      clearValues.color = {0.0f, 0.0f, 0.0f, 0.f};

      VkRenderPassBeginInfo renderPassBeginInfo =
          vks::initializers::renderPassBeginInfo();
      // renderPassBeginInfo.renderPass = renderPass_;
      // renderPassBeginInfo.framebuffer = frameBuffers_[currentImageIndex_];
      renderPassBeginInfo.renderPass = offscreenPass_.renderPass;

      const VulkanExample::FrameBuffer& fb = offscreenPass_.original;
      renderPassBeginInfo.framebuffer = fb.framebuffer;
      renderPassBeginInfo.renderArea.extent.width = fb.width;
      renderPassBeginInfo.renderArea.extent.height = fb.height;
      renderPassBeginInfo.clearValueCount = 1;
      renderPassBeginInfo.pClearValues = &clearValues;

      vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                           VK_SUBPASS_CONTENTS_INLINE);

      VkViewport viewport = vks::initializers::viewport(
          (float)fb.width, (float)fb.height, 0.0f, 1.0f);
      vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

      VkRect2D scissor = vks::initializers::rect2D(fb.width, fb.height, 0, 0);
      vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

      vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipelineLayouts_.blackhole, 0, 1,
                              &descriptorSets_[currentBuffer_].blackhole, 0,
                              nullptr);

      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipelines_.blackhole);
      vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

      vkCmdEndRenderPass(cmdBuffer);
    }

    // Brightness
    {
      VkClearValue clearValues{};
      clearValues.color = {0.f, 0.0f, 0.0f, 0.f};

      VkRenderPassBeginInfo renderPassBeginInfo =
          vks::initializers::renderPassBeginInfo();
      renderPassBeginInfo.renderPass = offscreenPass_.renderPass;
      const VulkanExample::FrameBuffer& fb = offscreenPass_.brightness;
      renderPassBeginInfo.framebuffer = fb.framebuffer;
      renderPassBeginInfo.renderArea.offset.x = 0;
      renderPassBeginInfo.renderArea.offset.y = 0;
      renderPassBeginInfo.renderArea.extent.width = fb.width;
      renderPassBeginInfo.renderArea.extent.height = fb.height;
      renderPassBeginInfo.clearValueCount = 1;
      renderPassBeginInfo.pClearValues = &clearValues;

      vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                           VK_SUBPASS_CONTENTS_INLINE);
      VkViewport viewport = vks::initializers::viewport(
          (float)fb.width, (float)fb.height, 0.0f, 1.0f);
      vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

      VkRect2D scissor = vks::initializers::rect2D(fb.width, fb.height, 0, 0);
      vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

      vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipelineLayouts_.brightness, 0, 1,
                              &descriptorSets_[currentBuffer_].brightness, 0,
                              nullptr);

      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipelines_.brightness);
      vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

      vkCmdEndRenderPass(cmdBuffer);
    }

    // Down sampling
    downSamplingCmdBuffer(cmdBuffer);

    // Up sampling
    upSamplingCmdBuffer(cmdBuffer);

    // Blend
    {
      VkClearValue clearValues{};
      clearValues.color = {0.f, 0.0f, 0.0f, 0.f};

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
      VkViewport viewport = vks::initializers::viewport(
          (float)width_, (float)height_, 0.0f, 1.0f);
      vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

      VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
      vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

      vkCmdBindDescriptorSets(
          cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.blend, 0,
          1, &descriptorSets_[currentBuffer_].blend, 0, nullptr);

      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipelines_.blend);
      vkCmdDraw(cmdBuffer, 6, 1, 0, 0);
      drawUI(cmdBuffer);
      vkCmdEndRenderPass(cmdBuffer);
    }
    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void downSamplingCmdBuffer(VkCommandBuffer& cmdBuffer) {
    VkClearValue clearValues{};
    clearValues.color = {0.f, 0.0f, 0.0f, 0.f};

    VkRenderPassBeginInfo renderPassBeginInfo =
        vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = offscreenPass_.renderPass;

    for (int sample_level = 0; sample_level < NUM_SAMPLE_SIZES;
         sample_level++) {
      // (1) Update and copy uniforms
      ubos_.downsample.srcResolution =
          glm::vec2(offscreenPass_.down_samples[sample_level].width,
                    offscreenPass_.down_samples[sample_level].height);
      memcpy(uniformBuffers_[currentBuffer_].downsample[sample_level].mapped,
             &ubos_.downsample, sizeof(DownsampleUBO));

      // (2) Set render fb target
      const VulkanExample::FrameBuffer& fb =
          offscreenPass_.down_samples[sample_level];
      renderPassBeginInfo.framebuffer = fb.framebuffer;
      renderPassBeginInfo.renderArea.extent.width = fb.width;
      renderPassBeginInfo.renderArea.extent.height = fb.height;
      renderPassBeginInfo.clearValueCount = 1;
      renderPassBeginInfo.pClearValues = &clearValues;

      // (3) Render
      vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                           VK_SUBPASS_CONTENTS_INLINE);

      VkViewport viewport = vks::initializers::viewport(
          (float)fb.width, (float)fb.height, 0.0f, 1.0f);
      vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

      VkRect2D scissor = vks::initializers::rect2D(fb.width, fb.height, 0, 0);
      vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

      vkCmdBindDescriptorSets(
          cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
          pipelineLayouts_.downsample, 0, 1,
          &descriptorSets_[currentBuffer_].downsamples[sample_level], 0,
          nullptr);

      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipelines_.downsample);
      vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

      vkCmdEndRenderPass(cmdBuffer);
    }
  }

  void upSamplingCmdBuffer(VkCommandBuffer& cmdBuffer) {
    VkClearValue clearValues{};
    clearValues.color = {0.f, 0.0f, 0.0f, 0.0f};

    VkRenderPassBeginInfo renderPassBeginInfo =
        vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = offscreenPass_.renderPass;

    for (int sample_level = NUM_SAMPLE_SIZES - 1; sample_level >= 0;
         sample_level--) {
      ubos_.upsample.dstResolution =
          glm::vec2(offscreenPass_.up_samples[sample_level].width,
                    offscreenPass_.up_samples[sample_level].height);
      memcpy(uniformBuffers_[currentBuffer_].upsample[sample_level].mapped,
             &ubos_.upsample, sizeof(DownsampleUBO));

      // Set render fb target
      const VulkanExample::FrameBuffer& fb =
          offscreenPass_.up_samples[sample_level];
      renderPassBeginInfo.framebuffer = fb.framebuffer;
      renderPassBeginInfo.renderArea.extent.width = fb.width;
      renderPassBeginInfo.renderArea.extent.height = fb.height;
      renderPassBeginInfo.clearValueCount = 1;
      renderPassBeginInfo.pClearValues = &clearValues;

      // Render
      vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                           VK_SUBPASS_CONTENTS_INLINE);

      VkViewport viewport = vks::initializers::viewport(
          (float)fb.width, (float)fb.height, 0.0f, 1.0f);
      vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

      VkRect2D scissor = vks::initializers::rect2D(fb.width, fb.height, 0, 0);
      vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

      vkCmdBindDescriptorSets(
          cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.upsample,
          0, 1, &descriptorSets_[currentBuffer_].upsamples[sample_level], 0,
          nullptr);

      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipelines_.upsample);
      vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

      vkCmdEndRenderPass(cmdBuffer);
    }
  }

  virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay) {
    if (overlay->header("Settings")) {
      overlay->checkBox("Show Blackhole", &showBlackholeUI);
      overlay->checkBox("Show Gravitational Lensing",
                        &gravatationalLensingEnabled);
      overlay->checkBox("Accretion Disk Enabled", &accDiskEnabled);
      overlay->checkBox("Disk Particles Enabled", &accDiskParticleEnabled);
      overlay->sliderFloat("Accretion Disk Height",
                           &ubos_.blackhole.accDiskHeight, 0.0, 1.0);
      overlay->sliderFloat("Accretion Disk Intensity",
                           &ubos_.blackhole.accDiskLit, 0.0, 1.0);
      overlay->sliderFloat("Accretion Disk Density V",
                           &ubos_.blackhole.accDiskDensityV, 0.0, 3.0);
      overlay->sliderFloat("Accretion Disk Density H",
                           &ubos_.blackhole.accDiskDensityH, 0.0, 5.0);
      overlay->sliderFloat("Accretion Disk Noise Scale",
                           &ubos_.blackhole.accDiskNoiseScale, 0.0f, 5.0f);
      overlay->sliderInt("Accretion Disk LOD", &ubos_.blackhole.accDiskNoiseLOD,
                         0, 10);
      overlay->sliderFloat("Accretion Disk Speed",
                           &ubos_.blackhole.accDiskSpeed, 0.0, 2.0);
      overlay->checkBox("Tone Mapping Enabled", &toneMappingEnabled);
      overlay->sliderFloat("Exposure", &ubos_.blend.exposure, 0.1f, 4.0f);
      overlay->sliderFloat("Bloom Strength", &ubos_.blend.bloomStrength, 0.0f,
                           1.0f);
    }
  }

  void loadTexture(const std::string& filename) {
    // We use the Khronos texture format
    // (https://www.khronos.org/opengles/sdk/tools/KTX/file_format_spec/)
    // Texture data contains 4 channels (RGBA) with unnormalized 8-bit values,
    // this is the most commonly supported format
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

    ktxResult result;
    ktxTexture* ktxTexture;

    if (!vks::tools::fileExists(filename)) {
      vks::tools::exitFatal("Could not load texture from " + filename +
                                "\n\nMake sure the assets submodule has been "
                                "checked out and is up-to-date.",
                            -1);
    }
    result = ktxTexture_CreateFromNamedFile(
        filename.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
    assert(result == KTX_SUCCESS);

    // Get properties required for using and upload texture data from the ktx
    // texture object
    accretionDiskTextureMap_.width = ktxTexture->baseWidth;
    accretionDiskTextureMap_.height = ktxTexture->baseHeight;
    accretionDiskTextureMap_.mipLevels = ktxTexture->numLevels;
    ktx_uint8_t* ktxTextureData = ktxTexture_GetData(ktxTexture);
    ktx_size_t ktxTextureSize = ktxTexture_GetSize(ktxTexture);

    VkMemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs = {};

    // Copy data to an optimal tiled image
    // This loads the texture data into a host local buffer that is copied
    // to the optimal tiled image on the device

    // Create a host-visible staging buffer that contains the raw image data
    // This buffer will be the data source for copying texture data to the
    // optimal tiled image on the device
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    VkBufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo();
    bufferCreateInfo.size = ktxTextureSize;
    // This buffer is used as a transfer source for the buffer copy
    bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK_RESULT(
        vkCreateBuffer(device_, &bufferCreateInfo, nullptr, &stagingBuffer));

    // Get memory requirements for the staging buffer (alignment, memory
    // type bits)
    vkGetBufferMemoryRequirements(device_, stagingBuffer, &memReqs);
    memAllocInfo.allocationSize = memReqs.size;
    // Get memory type index for a host visible buffer
    memAllocInfo.memoryTypeIndex = vulkanDevice_->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK_RESULT(
        vkAllocateMemory(device_, &memAllocInfo, nullptr, &stagingMemory));
    VK_CHECK_RESULT(
        vkBindBufferMemory(device_, stagingBuffer, stagingMemory, 0));

    // Copy texture data into host local staging buffer
    uint8_t* data;
    VK_CHECK_RESULT(
        vkMapMemory(device_, stagingMemory, 0, memReqs.size, 0, (void**)&data));
    memcpy(data, ktxTextureData, ktxTextureSize);
    vkUnmapMemory(device_, stagingMemory);

    // Setup buffer copy regions for each mip level
    std::vector<VkBufferImageCopy> bufferCopyRegions;
    uint32_t offset = 0;

    for (uint32_t i = 0; i < accretionDiskTextureMap_.mipLevels; i++) {
      // Calculate offset into staging buffer for the current mip level
      ktx_size_t offset;
      KTX_error_code ret =
          ktxTexture_GetImageOffset(ktxTexture, i, 0, 0, &offset);
      assert(ret == KTX_SUCCESS);
      // Setup a buffer image copy structure for the current mip level
      VkBufferImageCopy bufferCopyRegion = {};
      bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      bufferCopyRegion.imageSubresource.mipLevel = i;
      bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
      bufferCopyRegion.imageSubresource.layerCount = 1;
      bufferCopyRegion.imageExtent.width = ktxTexture->baseWidth >> i;
      bufferCopyRegion.imageExtent.height = ktxTexture->baseHeight >> i;
      bufferCopyRegion.imageExtent.depth = 1;
      bufferCopyRegion.bufferOffset = offset;
      bufferCopyRegions.push_back(bufferCopyRegion);
    }

    // Create optimal tiled target image on the device
    VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = format;
    imageCreateInfo.mipLevels = accretionDiskTextureMap_.mipLevels;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    // Set initial layout of the image to undefined
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.extent = {accretionDiskTextureMap_.width,
                              accretionDiskTextureMap_.height, 1};
    imageCreateInfo.usage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VK_CHECK_RESULT(vkCreateImage(device_, &imageCreateInfo, nullptr,
                                  &accretionDiskTextureMap_.image));

    vkGetImageMemoryRequirements(device_, accretionDiskTextureMap_.image,
                                 &memReqs);
    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex = vulkanDevice_->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device_, &memAllocInfo, nullptr,
                                     &accretionDiskTextureMap_.deviceMemory));
    VK_CHECK_RESULT(vkBindImageMemory(device_, accretionDiskTextureMap_.image,
                                      accretionDiskTextureMap_.deviceMemory,
                                      0));

    VkCommandBuffer copyCmd = vulkanDevice_->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    // Image memory barriers for the texture image

    // The sub resource range describes the regions of the image that will
    // be transitioned using the memory barriers below
    VkImageSubresourceRange subresourceRange = {};
    // Image only contains color data
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    // Start at first mip level
    subresourceRange.baseMipLevel = 0;
    // We will transition on all mip levels
    subresourceRange.levelCount = accretionDiskTextureMap_.mipLevels;
    // The 2D texture only has one layer
    subresourceRange.layerCount = 1;

    // Transition the texture image layout to transfer target, so we can
    // safely copy our buffer data to it.
    VkImageMemoryBarrier imageMemoryBarrier =
        vks::initializers::imageMemoryBarrier();
    ;
    imageMemoryBarrier.image = accretionDiskTextureMap_.image;
    imageMemoryBarrier.subresourceRange = subresourceRange;
    imageMemoryBarrier.srcAccessMask = 0;
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    // Insert a memory dependency at the proper pipeline stages that will
    // execute the image layout transition Source pipeline stage is host
    // write/read execution (VK_PIPELINE_STAGE_HOST_BIT) Destination
    // pipeline stage is copy command execution
    // (VK_PIPELINE_STAGE_TRANSFER_BIT)
    vkCmdPipelineBarrier(copyCmd, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &imageMemoryBarrier);

    // Copy mip levels from staging buffer
    vkCmdCopyBufferToImage(copyCmd, stagingBuffer,
                           accretionDiskTextureMap_.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           static_cast<uint32_t>(bufferCopyRegions.size()),
                           bufferCopyRegions.data());

    // Once the data has been uploaded we transfer to the texture image to
    // the shader read layout, so it can be sampled from
    imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Insert a memory dependency at the proper pipeline stages that will
    // execute the image layout transition Source pipeline stage is copy
    // command execution (VK_PIPELINE_STAGE_TRANSFER_BIT) Destination
    // pipeline stage fragment shader access
    // (VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
    vkCmdPipelineBarrier(copyCmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &imageMemoryBarrier);

    // Store current layout for later reuse
    accretionDiskTextureMap_.imageLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    vulkanDevice_->flushCommandBuffer(copyCmd, queue_, true);

    // Clean up staging resources
    vkFreeMemory(device_, stagingMemory, nullptr);
    vkDestroyBuffer(device_, stagingBuffer, nullptr);
    ktxTexture_Destroy(ktxTexture);

    // Create a texture sampler
    // In Vulkan textures are accessed by samplers
    // This separates all the sampling information from the texture data. This
    // means you could have multiple sampler objects for the same texture with
    // different settings Note: Similar to the samplers available with
    // OpenGL 3.3
    VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.mipLodBias = 0.0f;
    sampler.compareOp = VK_COMPARE_OP_NEVER;
    sampler.minLod = 0.0f;
    // Set max level-of-detail to mip level count of the texture
    sampler.maxLod = (float)accretionDiskTextureMap_.mipLevels;
    // Enable anisotropic filtering
    // This feature is optional, so we must check if it's supported on the
    // device
    if (vulkanDevice_->features.samplerAnisotropy) {
      // Use max. level of anisotropy for this example
      sampler.maxAnisotropy =
          vulkanDevice_->properties.limits.maxSamplerAnisotropy;
      sampler.anisotropyEnable = VK_TRUE;
    } else {
      // The device does not support anisotropic filtering
      sampler.maxAnisotropy = 1.0;
      sampler.anisotropyEnable = VK_FALSE;
    }
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(vkCreateSampler(device_, &sampler, nullptr,
                                    &accretionDiskTextureMap_.sampler));

    // Create image view
    // Textures are not directly accessed by the shaders and
    // are abstracted by image views containing additional
    // information and sub resource ranges
    VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    // The subresource range describes the set of mip levels (and array
    // layers) that can be accessed through this image view It's possible to
    // create multiple image views for a single image referring to different
    // (and/or overlapping) ranges of the image
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.baseMipLevel = 0;
    view.subresourceRange.baseArrayLayer = 0;
    view.subresourceRange.layerCount = 1;
    // Linear tiling usually won't support mip maps
    // Only set mip map count if optimal tiling is used
    view.subresourceRange.levelCount = accretionDiskTextureMap_.mipLevels;
    // The view will be based on the texture's image
    view.image = accretionDiskTextureMap_.image;
    VK_CHECK_RESULT(vkCreateImageView(device_, &view, nullptr,
                                      &accretionDiskTextureMap_.view));
  }

  // Enable physical device features required for this example
  virtual void getEnabledFeatures() {
    if (deviceFeatures_.samplerAnisotropy) {
      enabledFeatures_.samplerAnisotropy = VK_TRUE;
    }
  }

  void destroyOffscreenPass() {
    vkDestroyRenderPass(device_, offscreenPass_.renderPass, nullptr);
    vkDestroyFramebuffer(device_, offscreenPass_.original.framebuffer, nullptr);
    vkDestroyFramebuffer(device_, offscreenPass_.brightness.framebuffer,
                         nullptr);
    vkDestroyFramebuffer(device_, offscreenPass_.final.framebuffer, nullptr);
    for (FrameBuffer sample : offscreenPass_.down_samples) {
      vkDestroyFramebuffer(device_, sample.framebuffer, nullptr);
    }
  }

  void windowResized() override {
    destroyOffscreenPass();
    prepareOffscreen();
    vkResetDescriptorPool(device_, descriptorPool_, 0);
    setupDescriptors();
    resized_ = false;
  }

  VulkanExample() : VulkanExampleBase() {
    title = "Blackhole";
    camera_.type_ = Camera::CameraType::lookat;
    camera_.setPosition(glm::vec3(0.0f, 0.0f, -15.0f));
    camera_.setRotation(glm::vec3(0.0f));
    camera_.setRotationSpeed(0.25f);
    camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);
  }

  // (Part A.0) Called once in main() before renderLoop()
  void prepare() {
    VulkanExampleBase::prepare();
    loadAssets();
    prepareUniformBuffers();
    prepareOffscreen();
    setupDescriptors();
    preparePipelines();
    prepared_ = true;
  }

  // (A.1)
  void loadAssets() {
    uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices |
                                vkglTF::FileLoadingFlags::FlipY;
    // Cubemap texture
    cubeMap_.loadFromFile(
        getAssetPath() + "textures/blackhole/skybox/cubemap.ktx",
        VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice_, queue_);
    // loadTexture(getAssetPath() + "textures/blackhole/uv_checker.ktx");
    loadTexture(getAssetPath() + "textures/blackhole/blackhole_color_map.ktx");
  }

  ~VulkanExample() {
    if (device_) {
      vkDestroyImageView(device_, cubeMap_.view, nullptr);
      vkDestroyImage(device_, cubeMap_.image, nullptr);
      vkDestroySampler(device_, cubeMap_.sampler, nullptr);
      vkFreeMemory(device_, cubeMap_.deviceMemory, nullptr);
      vkDestroyPipeline(device_, pipelines_.blackhole, nullptr);
      vkDestroyPipeline(device_, pipelines_.blend, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.blackhole, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.blend, nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.blackhole,
                                   nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.downsample,
                                   nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.upsample,
                                   nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.blend,
                                   nullptr);
      vkDestroyImageView(device_, offscreenPass_.original.color.view, nullptr);
      vkDestroyImage(device_, offscreenPass_.original.color.image, nullptr);
      vkFreeMemory(device_, offscreenPass_.original.color.mem, nullptr);
      vkDestroyFramebuffer(device_, offscreenPass_.original.framebuffer,
                           nullptr);
      vkDestroyImageView(device_, offscreenPass_.brightness.color.view,
                         nullptr);
      vkDestroyImage(device_, offscreenPass_.brightness.color.image, nullptr);
      vkFreeMemory(device_, offscreenPass_.brightness.color.mem, nullptr);
      vkDestroyFramebuffer(device_, offscreenPass_.brightness.framebuffer,
                           nullptr);
      vkDestroyImageView(device_, offscreenPass_.final.color.view, nullptr);
      vkDestroyImage(device_, offscreenPass_.final.color.image, nullptr);
      vkFreeMemory(device_, offscreenPass_.final.color.mem, nullptr);
      vkDestroyFramebuffer(device_, offscreenPass_.final.framebuffer, nullptr);
      for (auto& sample : offscreenPass_.down_samples) {
        vkDestroyImageView(device_, sample.color.view, nullptr);
        vkDestroyImage(device_, sample.color.image, nullptr);
        vkFreeMemory(device_, sample.color.mem, nullptr);
        vkDestroyFramebuffer(device_, sample.framebuffer, nullptr);
      }
      for (auto& buffer : uniformBuffers_) {
        buffer.blackhole.destroy();
      }
    }
  }
};

VULKAN_EXAMPLE_MAIN()