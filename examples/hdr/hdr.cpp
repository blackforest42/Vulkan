/*
 * Vulkan Example - High dynamic range rendering pipeline
 *
 * This sample implements a HDR rendering pipeline that uses a wider range of
 * possible colors via float component image formats It also does a bloom filter
 * on the HDR image The final output is standard definition range (SDR) Note:
 * Does not make use of HDR display capability. HDR is only internally used for
 * offscreen rendering.
 *
 * Copyright (C) 2016-2025 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#include "VulkanglTFModel.h"
#include "vulkanexamplebase.h"

class VulkanExample : public VulkanExampleBase {
 public:
  bool bloom_ = true;
  bool displaySkybox_ = true;

  struct {
    vks::TextureCubeMap envmap;
  } textures_;

  struct Models {
    vkglTF::Model skybox;
    std::vector<vkglTF::Model> objects;
    int32_t index{1};
  } models_;
  std::vector<std::string> modelNames_{};

  struct UniformData {
    glm::mat4 projection;
    glm::mat4 modelview;
    glm::mat4 inverseModelview;
    float exposure{1.0f};
  } uniformData_;
  std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers_;

  struct {
    VkPipelineLayout models{VK_NULL_HANDLE};
    VkPipelineLayout composition{VK_NULL_HANDLE};
    VkPipelineLayout bloomFilter{VK_NULL_HANDLE};
  } pipelineLayouts_;

  struct {
    VkPipeline skybox{VK_NULL_HANDLE};
    VkPipeline reflect{VK_NULL_HANDLE};
    VkPipeline composition{VK_NULL_HANDLE};
    // Bloom is a two pass filter (one pass for vertical and horizontal blur)
    VkPipeline bloom[2]{VK_NULL_HANDLE};
  } pipelines_;

  struct {
    VkDescriptorSetLayout models{VK_NULL_HANDLE};
    VkDescriptorSetLayout composition{VK_NULL_HANDLE};
    VkDescriptorSetLayout bloomFilter{VK_NULL_HANDLE};
  } descriptorSetLayouts_;

  struct DescriptorSets {
    VkDescriptorSet object{VK_NULL_HANDLE};
    VkDescriptorSet skybox{VK_NULL_HANDLE};
    VkDescriptorSet composition{VK_NULL_HANDLE};
    VkDescriptorSet bloomFilter{VK_NULL_HANDLE};
  };
  std::array<DescriptorSets, MAX_CONCURRENT_FRAMES> descriptorSets_{};

  // Framebuffer for offscreen rendering
  struct FrameBufferAttachment {
    VkImage image;
    VkDeviceMemory mem;
    VkImageView view;
    VkFormat format;
    void destroy(VkDevice device) {
      vkDestroyImageView(device, view, nullptr);
      vkDestroyImage(device, image, nullptr);
      vkFreeMemory(device, mem, nullptr);
    }
  };
  struct FrameBuffer {
    int32_t width, height;
    VkFramebuffer frameBuffer;
    FrameBufferAttachment color[2];
    FrameBufferAttachment depth;
    VkRenderPass renderPass;
    VkSampler sampler;
  } offscreen_;

  struct {
    int32_t width, height;
    VkFramebuffer frameBuffer;
    FrameBufferAttachment color[1];
    VkRenderPass renderPass;
    VkSampler sampler;
  } filterPass_;

  VulkanExample() : VulkanExampleBase() {
    title_ = "High dynamic range rendering";
    camera_.type_ = Camera::CameraType::lookat;
    camera_.setPosition(glm::vec3(0.0f, 0.0f, -6.0f));
    camera_.setRotation(glm::vec3(0.0f, 0.0f, 0.0f));
    camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);
  }

  void createAttachment(VkFormat format,
                        VkImageUsageFlagBits usage,
                        FrameBufferAttachment* attachment) {
    VkImageAspectFlags aspectMask = 0;
    VkImageLayout imageLayout;

    attachment->format = format;

    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
      aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
      aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      if (format >= VK_FORMAT_D16_UNORM_S8_UINT)
        aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
      imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    assert(aspectMask > 0);

    VkImageCreateInfo image = vks::initializers::imageCreateInfo();
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = format;
    image.extent.width = offscreen_.width;
    image.extent.height = offscreen_.height;
    image.extent.depth = 1;
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = usage | VK_IMAGE_USAGE_SAMPLED_BIT;

    VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs;

    VK_CHECK_RESULT(
        vkCreateImage(device_, &image, nullptr, &attachment->image));
    vkGetImageMemoryRequirements(device_, attachment->image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = vulkanDevice_->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(
        vkAllocateMemory(device_, &memAlloc, nullptr, &attachment->mem));
    VK_CHECK_RESULT(
        vkBindImageMemory(device_, attachment->image, attachment->mem, 0));

    VkImageViewCreateInfo imageView = vks::initializers::imageViewCreateInfo();
    imageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageView.format = format;
    imageView.subresourceRange = {};
    imageView.subresourceRange.aspectMask = aspectMask;
    imageView.subresourceRange.baseMipLevel = 0;
    imageView.subresourceRange.levelCount = 1;
    imageView.subresourceRange.baseArrayLayer = 0;
    imageView.subresourceRange.layerCount = 1;
    imageView.image = attachment->image;
    VK_CHECK_RESULT(
        vkCreateImageView(device_, &imageView, nullptr, &attachment->view));
  }

  void prepare() {
    VulkanExampleBase::prepare();
    loadAssets();
    prepareUniformBuffers();
    prepareoffscreenfer();
    setupDescriptors();
    preparePipelines();
    prepared_ = true;
  }

  void loadAssets() {
    // Load glTF models
    const uint32_t glTFLoadingFlags =
        vkglTF::FileLoadingFlags::PreTransformVertices |
        vkglTF::FileLoadingFlags::FlipY;
    models_.skybox.loadFromFile(getAssetPath() + "models/cube.gltf",
                                vulkanDevice_, queue_, glTFLoadingFlags);
    std::vector<std::string> filenames = {"sphere.gltf", "teapot.gltf",
                                          "torusknot.gltf", "venus.gltf"};
    modelNames_ = {"Sphere", "Teapot", "Torusknot", "Venus"};
    models_.objects.resize(filenames.size());
    for (size_t i = 0; i < filenames.size(); i++) {
      models_.objects[i].loadFromFile(getAssetPath() + "models/" + filenames[i],
                                      vulkanDevice_, queue_, glTFLoadingFlags);
    }
    // Load HDR cube map
    textures_.envmap.loadFromFile(
        getAssetPath() + "textures/hdr/uffizi_cube.ktx",
        VK_FORMAT_R16G16B16A16_SFLOAT, vulkanDevice_, queue_);
  }

  // Prepare and initialize uniform buffer containing shader uniforms
  void prepareUniformBuffers() {
    for (auto& buffer : uniformBuffers_) {
      // Scene matrices uniform buffer
      VK_CHECK_RESULT(
          vulkanDevice_->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      &buffer, sizeof(UniformData)));
      VK_CHECK_RESULT(buffer.map());
    }
  }

  // Prepare a new framebuffer and attachments for offscreen rendering
  // (G-Buffer)
  void prepareoffscreenfer() {
    {
      offscreen_.width = width_;
      offscreen_.height = height_;

      // Color attachments

      // Two floating point color buffers
      createAttachment(VK_FORMAT_R32G32B32A32_SFLOAT,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                       &offscreen_.color[0]);
      createAttachment(VK_FORMAT_R32G32B32A32_SFLOAT,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                       &offscreen_.color[1]);
      // Depth attachment
      createAttachment(depthFormat_,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       &offscreen_.depth);

      // Set up separate renderpass with references to the color and depth
      // attachments
      std::array<VkAttachmentDescription, 3> attachmentDescs = {};

      // Init attachment properties
      for (uint32_t i = 0; i < 3; ++i) {
        attachmentDescs[i].samples = VK_SAMPLE_COUNT_1_BIT;
        attachmentDescs[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachmentDescs[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachmentDescs[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescs[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        if (i == 2) {
          attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
          attachmentDescs[i].finalLayout =
              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        } else {
          attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
          attachmentDescs[i].finalLayout =
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
      }

      // Formats
      attachmentDescs[0].format = offscreen_.color[0].format;
      attachmentDescs[1].format = offscreen_.color[1].format;
      attachmentDescs[2].format = offscreen_.depth.format;

      std::vector<VkAttachmentReference> colorReferences;
      colorReferences.push_back({0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
      colorReferences.push_back({1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

      VkAttachmentReference depthReference = {};
      depthReference.attachment = 2;
      depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

      VkSubpassDescription subpass = {};
      subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      subpass.pColorAttachments = colorReferences.data();
      subpass.colorAttachmentCount = 2;
      subpass.pDepthStencilAttachment = &depthReference;

      // Use subpass dependencies for attachment layout transitions
      std::array<VkSubpassDependency, 3> dependencies{};

      dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
      dependencies[0].dstSubpass = 0;
      dependencies[0].srcStageMask =
          VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
          VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
      dependencies[0].dstStageMask =
          VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
          VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
      dependencies[0].srcAccessMask =
          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      dependencies[0].dstAccessMask =
          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
      dependencies[0].dependencyFlags = 0;

      dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
      dependencies[1].dstSubpass = 0;
      dependencies[1].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
      dependencies[1].dstStageMask =
          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependencies[1].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
      dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

      dependencies[2].srcSubpass = 0;
      dependencies[2].dstSubpass = VK_SUBPASS_EXTERNAL;
      dependencies[2].srcStageMask =
          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependencies[2].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
      dependencies[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      dependencies[2].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
      dependencies[2].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

      VkRenderPassCreateInfo renderPassInfo = {};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
      renderPassInfo.pAttachments = attachmentDescs.data();
      renderPassInfo.attachmentCount =
          static_cast<uint32_t>(attachmentDescs.size());
      renderPassInfo.subpassCount = 1;
      renderPassInfo.pSubpasses = &subpass;
      renderPassInfo.dependencyCount =
          static_cast<uint32_t>(dependencies.size());
      renderPassInfo.pDependencies = dependencies.data();

      VK_CHECK_RESULT(vkCreateRenderPass(device_, &renderPassInfo, nullptr,
                                         &offscreen_.renderPass));

      std::array<VkImageView, 3> attachments;
      attachments[0] = offscreen_.color[0].view;
      attachments[1] = offscreen_.color[1].view;
      attachments[2] = offscreen_.depth.view;

      VkFramebufferCreateInfo fbufCreateInfo = {};
      fbufCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      fbufCreateInfo.pNext = NULL;
      fbufCreateInfo.renderPass = offscreen_.renderPass;
      fbufCreateInfo.pAttachments = attachments.data();
      fbufCreateInfo.attachmentCount =
          static_cast<uint32_t>(attachments.size());
      fbufCreateInfo.width = offscreen_.width;
      fbufCreateInfo.height = offscreen_.height;
      fbufCreateInfo.layers = 1;
      VK_CHECK_RESULT(vkCreateFramebuffer(device_, &fbufCreateInfo, nullptr,
                                          &offscreen_.frameBuffer));

      // Create sampler to sample from the color attachments
      VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
      sampler.magFilter = VK_FILTER_NEAREST;
      sampler.minFilter = VK_FILTER_NEAREST;
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
          vkCreateSampler(device_, &sampler, nullptr, &offscreen_.sampler));
    }

    // Bloom separable filter pass
    {
      filterPass_.width = width_;
      filterPass_.height = height_;

      // Color attachments

      // Two floating point color buffers
      createAttachment(VK_FORMAT_R32G32B32A32_SFLOAT,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                       &filterPass_.color[0]);

      // Set up separate renderpass with references to the color and depth
      // attachments
      std::array<VkAttachmentDescription, 1> attachmentDescs = {};

      // Init attachment properties
      attachmentDescs[0].samples = VK_SAMPLE_COUNT_1_BIT;
      attachmentDescs[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      attachmentDescs[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      attachmentDescs[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      attachmentDescs[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      attachmentDescs[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      attachmentDescs[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      attachmentDescs[0].format = filterPass_.color[0].format;

      std::vector<VkAttachmentReference> colorReferences;
      colorReferences.push_back({0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

      VkSubpassDescription subpass = {};
      subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      subpass.pColorAttachments = colorReferences.data();
      subpass.colorAttachmentCount = 1;

      // Use subpass dependencies for attachment layout transitions
      std::array<VkSubpassDependency, 2> dependencies;

      dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
      dependencies[0].dstSubpass = 0;
      dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
      dependencies[0].dstStageMask =
          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
      dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

      dependencies[1].srcSubpass = 0;
      dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
      dependencies[1].srcStageMask =
          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
      dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
      dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

      VkRenderPassCreateInfo renderPassInfo = {};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
      renderPassInfo.pAttachments = attachmentDescs.data();
      renderPassInfo.attachmentCount =
          static_cast<uint32_t>(attachmentDescs.size());
      renderPassInfo.subpassCount = 1;
      renderPassInfo.pSubpasses = &subpass;
      renderPassInfo.dependencyCount = 2;
      renderPassInfo.pDependencies = dependencies.data();

      VK_CHECK_RESULT(vkCreateRenderPass(device_, &renderPassInfo, nullptr,
                                         &filterPass_.renderPass));

      std::array<VkImageView, 1> attachments;
      attachments[0] = filterPass_.color[0].view;

      VkFramebufferCreateInfo fbufCreateInfo = {};
      fbufCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      fbufCreateInfo.pNext = NULL;
      fbufCreateInfo.renderPass = filterPass_.renderPass;
      fbufCreateInfo.pAttachments = attachments.data();
      fbufCreateInfo.attachmentCount =
          static_cast<uint32_t>(attachments.size());
      fbufCreateInfo.width = filterPass_.width;
      fbufCreateInfo.height = filterPass_.height;
      fbufCreateInfo.layers = 1;
      VK_CHECK_RESULT(vkCreateFramebuffer(device_, &fbufCreateInfo, nullptr,
                                          &filterPass_.frameBuffer));

      // Create sampler to sample from the color attachments
      VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
      sampler.magFilter = VK_FILTER_NEAREST;
      sampler.minFilter = VK_FILTER_NEAREST;
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
          vkCreateSampler(device_, &sampler, nullptr, &filterPass_.sampler));
    }
  }

  void setupDescriptors() {
    // Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              MAX_CONCURRENT_FRAMES * 2),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            MAX_CONCURRENT_FRAMES * 6)};
    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(
            static_cast<uint32_t>(poolSizes.size()), poolSizes.data(),
            MAX_CONCURRENT_FRAMES * 4);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo,
                                           nullptr, &descriptorPool_));

    // Layouts
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0),
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, 1),
    };
    VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo =
        vks::initializers::descriptorSetLayoutCreateInfo(
            setLayoutBindings.data(),
            static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorLayoutInfo,
                                                nullptr,
                                                &descriptorSetLayouts_.models));

    // Bloom filter
    setLayoutBindings = {
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, 0),
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, 1),
    };

    descriptorLayoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorLayoutInfo, nullptr,
                                    &descriptorSetLayouts_.bloomFilter));

    // G-Buffer composition
    setLayoutBindings = {
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, 0),
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, 1),
    };

    descriptorLayoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorLayoutInfo, nullptr,
                                    &descriptorSetLayouts_.composition));

    // Sets per frame, just like the buffers themselves
    // Images do not need to be duplicated per frame, we reuse the same one for
    // each frame
    for (auto i = 0; i < uniformBuffers_.size(); i++) {
      // 3D object descriptor set
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool_, &descriptorSetLayouts_.models, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].object));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].object, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
              &uniformBuffers_[i].descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].object,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
              &textures_.envmap.descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Sky box descriptor set
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].skybox));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].skybox, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
              &uniformBuffers_[i].descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].skybox,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
              &textures_.envmap.descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Bloom filter
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.bloomFilter, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &descriptorSets_[i].bloomFilter));
      std::vector<VkDescriptorImageInfo> colorDescriptors = {
          vks::initializers::descriptorImageInfo(
              offscreen_.sampler, offscreen_.color[0].view,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
          vks::initializers::descriptorImageInfo(
              offscreen_.sampler, offscreen_.color[1].view,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
      };
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].bloomFilter,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0,
              &colorDescriptors[0]),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].bloomFilter,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
              &colorDescriptors[1]),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Composition descriptor set
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.composition, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &descriptorSets_[i].composition));
      colorDescriptors = {
          vks::initializers::descriptorImageInfo(
              offscreen_.sampler, offscreen_.color[0].view,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
          vks::initializers::descriptorImageInfo(
              offscreen_.sampler, filterPass_.color[0].view,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
      };
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].composition,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0,
              &colorDescriptors[0]),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].composition,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
              &colorDescriptors[1]),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  void preparePipelines() {
    // Layouts
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &descriptorSetLayouts_.models, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo,
                                           nullptr, &pipelineLayouts_.models));

    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.bloomFilter, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo,
                                           nullptr,
                                           &pipelineLayouts_.bloomFilter));

    pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.composition, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo,
                                           nullptr,
                                           &pipelineLayouts_.composition));

    // Pipelines
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
        vks::initializers::pipelineInputAssemblyStateCreateInfo(
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
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
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

    VkGraphicsPipelineCreateInfo pipelineCI =
        vks::initializers::pipelineCreateInfo(pipelineLayouts_.models,
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

    VkSpecializationInfo specializationInfo;
    std::array<VkSpecializationMapEntry, 1> specializationMapEntries{};

    // Full screen pipelines

    // Empty vertex input state, full screen triangles are generated by the
    // vertex shader
    VkPipelineVertexInputStateCreateInfo emptyInputState =
        vks::initializers::pipelineVertexInputStateCreateInfo();
    pipelineCI.pVertexInputState = &emptyInputState;

    // Final fullscreen composition pass pipeline
    std::vector<VkPipelineColorBlendAttachmentState> blendAttachmentStates = {
        vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
        vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
    };
    pipelineCI.layout = pipelineLayouts_.composition;
    pipelineCI.renderPass = renderPass_;
    rasterizationState.cullMode = VK_CULL_MODE_NONE;
    colorBlendState.attachmentCount = 1;
    colorBlendState.pAttachments = blendAttachmentStates.data();
    shaderStages[0] = loadShader(getShadersPath() + "hdr/composition.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "hdr/composition.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.composition));

    // Bloom pass
    shaderStages[0] = loadShader(getShadersPath() + "hdr/bloom.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "hdr/bloom.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    colorBlendState.pAttachments = &blendAttachmentState;
    blendAttachmentState.colorWriteMask = 0xF;
    blendAttachmentState.blendEnable = VK_TRUE;
    blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;

    // Set constant parameters via specialization constants
    specializationMapEntries[0] =
        vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
    uint32_t dir = 1;
    specializationInfo = vks::initializers::specializationInfo(
        1, specializationMapEntries.data(), sizeof(dir), &dir);
    shaderStages[1].pSpecializationInfo = &specializationInfo;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.bloom[0]));

    // Second blur pass (into separate framebuffer)
    pipelineCI.renderPass = filterPass_.renderPass;
    dir = 0;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.bloom[1]));

    // Object rendering pipelines
    // Use vertex input state from glTF model setup
    pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState(
        {vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal});

    blendAttachmentState.blendEnable = VK_FALSE;
    pipelineCI.layout = pipelineLayouts_.models;
    pipelineCI.renderPass = offscreen_.renderPass;
    colorBlendState.attachmentCount = 2;
    colorBlendState.pAttachments = blendAttachmentStates.data();
    shaderStages[0] = loadShader(getShadersPath() + "hdr/gbuffer.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(getShadersPath() + "hdr/gbuffer.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    // Set constant parameters via specialization constants
    specializationMapEntries[0] =
        vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
    uint32_t shadertype = 0;
    specializationInfo = vks::initializers::specializationInfo(
        1, specializationMapEntries.data(), sizeof(shadertype), &shadertype);
    shaderStages[0].pSpecializationInfo = &specializationInfo;
    shaderStages[1].pSpecializationInfo = &specializationInfo;
    // Skybox pipeline (background cube)
    rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.skybox));

    // Object rendering pipeline
    shadertype = 1;
    // Enable depth test and write
    depthStencilState.depthWriteEnable = VK_TRUE;
    depthStencilState.depthTestEnable = VK_TRUE;
    // Flip cull mode
    rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.reflect));
  }

  virtual void render() {
    if (!prepared_)
      return;
    VulkanExampleBase::prepareFrame();
    updateUniformBuffers();
    buildCommandBuffer();
    VulkanExampleBase::submitFrame();
  }

  void updateUniformBuffers() {
    uniformData_.projection = camera_.matrices_.perspective;
    uniformData_.modelview = camera_.matrices_.view;
    uniformData_.inverseModelview = glm::inverse(camera_.matrices_.view);
    memcpy(uniformBuffers_[currentBuffer_].mapped, &uniformData_,
           sizeof(uniformData_));
  }

  void buildCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();

    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    {
      /*
              First pass: Render scene to offscreen framebuffer
      */

      std::array<VkClearValue, 3> clearValues{};
      clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
      clearValues[1].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
      clearValues[2].depthStencil = {1.0f, 0};

      VkRenderPassBeginInfo renderPassBeginInfo =
          vks::initializers::renderPassBeginInfo();
      renderPassBeginInfo.renderPass = offscreen_.renderPass;
      renderPassBeginInfo.framebuffer = offscreen_.frameBuffer;
      renderPassBeginInfo.renderArea.extent.width = offscreen_.width;
      renderPassBeginInfo.renderArea.extent.height = offscreen_.height;
      renderPassBeginInfo.clearValueCount = 3;
      renderPassBeginInfo.pClearValues = clearValues.data();

      vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                           VK_SUBPASS_CONTENTS_INLINE);

      VkViewport viewport = vks::initializers::viewport(
          (float)offscreen_.width, (float)offscreen_.height, 0.0f, 1.0f);
      vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

      VkRect2D scissor =
          vks::initializers::rect2D(offscreen_.width, offscreen_.height, 0, 0);
      vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

      VkDeviceSize offsets[1] = {0};

      // Skybox
      if (displaySkybox_) {
        vkCmdBindDescriptorSets(
            cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.models,
            0, 1, &descriptorSets_[currentBuffer_].skybox, 0, nullptr);
        vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &models_.skybox.vertices.buffer,
                               offsets);
        vkCmdBindIndexBuffer(cmdBuffer, models_.skybox.indices.buffer, 0,
                             VK_INDEX_TYPE_UINT32);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelines_.skybox);
        models_.skybox.draw(cmdBuffer);
      }

      // 3D object
      vkCmdBindDescriptorSets(
          cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.models,
          0, 1, &descriptorSets_[currentBuffer_].object, 0, nullptr);
      vkCmdBindVertexBuffers(cmdBuffer, 0, 1,
                             &models_.objects[models_.index].vertices.buffer,
                             offsets);
      vkCmdBindIndexBuffer(cmdBuffer,
                           models_.objects[models_.index].indices.buffer, 0,
                           VK_INDEX_TYPE_UINT32);
      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipelines_.reflect);
      models_.objects[models_.index].draw(cmdBuffer);

      vkCmdEndRenderPass(cmdBuffer);
    }

    /*
            Second render pass: First bloom pass
    */
    if (bloom_) {
      VkClearValue clearValues[2]{};
      clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
      clearValues[1].depthStencil = {1.0f, 0};

      // Bloom filter
      VkRenderPassBeginInfo renderPassBeginInfo =
          vks::initializers::renderPassBeginInfo();
      renderPassBeginInfo.framebuffer = filterPass_.frameBuffer;
      renderPassBeginInfo.renderPass = filterPass_.renderPass;
      renderPassBeginInfo.clearValueCount = 1;
      renderPassBeginInfo.renderArea.extent.width = filterPass_.width;
      renderPassBeginInfo.renderArea.extent.height = filterPass_.height;
      renderPassBeginInfo.pClearValues = clearValues;

      vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                           VK_SUBPASS_CONTENTS_INLINE);

      VkViewport viewport = vks::initializers::viewport(
          (float)filterPass_.width, (float)filterPass_.height, 0.0f, 1.0f);
      vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

      VkRect2D scissor = vks::initializers::rect2D(filterPass_.width,
                                                   filterPass_.height, 0, 0);
      vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

      vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipelineLayouts_.bloomFilter, 0, 1,
                              &descriptorSets_[currentBuffer_].bloomFilter, 0,
                              nullptr);

      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipelines_.bloom[1]);
      vkCmdDraw(cmdBuffer, 3, 1, 0, 0);

      vkCmdEndRenderPass(cmdBuffer);
    }

    /*
            Note: Explicit synchronization is not required between the render
       pass, as this is done implicit via sub pass dependencies
    */

    /*
            Third render pass: Scene rendering with applied second bloom pass
       (when enabled)
    */
    {
      VkClearValue clearValues[2]{};
      clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
      clearValues[1].depthStencil = {1.0f, 0};

      // Final composition
      VkRenderPassBeginInfo renderPassBeginInfo =
          vks::initializers::renderPassBeginInfo();
      renderPassBeginInfo.framebuffer = frameBuffers_[currentImageIndex_];
      renderPassBeginInfo.renderPass = renderPass_;
      renderPassBeginInfo.clearValueCount = 2;
      renderPassBeginInfo.renderArea.extent.width = width_;
      renderPassBeginInfo.renderArea.extent.height = height_;
      renderPassBeginInfo.pClearValues = clearValues;

      vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                           VK_SUBPASS_CONTENTS_INLINE);

      VkViewport viewport = vks::initializers::viewport(
          (float)width_, (float)height_, 0.0f, 1.0f);
      vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

      VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
      vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

      vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipelineLayouts_.composition, 0, 1,
                              &descriptorSets_[currentBuffer_].composition, 0,
                              nullptr);

      // Scene
      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipelines_.composition);
      vkCmdDraw(cmdBuffer, 3, 1, 0, 0);

      // Bloom
      if (bloom_) {
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelines_.bloom[0]);
        vkCmdDraw(cmdBuffer, 3, 1, 0, 0);
      }

      drawUI(cmdBuffer);

      vkCmdEndRenderPass(cmdBuffer);
    }

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  ~VulkanExample() {
    if (device_) {
      vkDestroyPipeline(device_, pipelines_.skybox, nullptr);
      vkDestroyPipeline(device_, pipelines_.reflect, nullptr);
      vkDestroyPipeline(device_, pipelines_.composition, nullptr);
      vkDestroyPipeline(device_, pipelines_.bloom[0], nullptr);
      vkDestroyPipeline(device_, pipelines_.bloom[1], nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.models, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.composition, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.bloomFilter, nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.models,
                                   nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.composition,
                                   nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.bloomFilter,
                                   nullptr);
      vkDestroyRenderPass(device_, offscreen_.renderPass, nullptr);
      vkDestroyRenderPass(device_, filterPass_.renderPass, nullptr);
      vkDestroyFramebuffer(device_, offscreen_.frameBuffer, nullptr);
      vkDestroyFramebuffer(device_, filterPass_.frameBuffer, nullptr);
      vkDestroySampler(device_, offscreen_.sampler, nullptr);
      vkDestroySampler(device_, filterPass_.sampler, nullptr);
      offscreen_.depth.destroy(device_);
      offscreen_.color[0].destroy(device_);
      offscreen_.color[1].destroy(device_);
      filterPass_.color[0].destroy(device_);
      textures_.envmap.destroy();
      for (auto& buffer : uniformBuffers_) {
        buffer.destroy();
      }
    }
  }

  virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay) {
    if (overlay->header("Settings")) {
      overlay->comboBox("Object type", &models_.index, modelNames_);
      overlay->inputFloat("Exposure", &uniformData_.exposure, 0.025f, 3);
      overlay->checkBox("Bloom", &bloom_);
      overlay->checkBox("Skybox", &displaySkybox_);
    }
  }
};

VULKAN_EXAMPLE_MAIN()
