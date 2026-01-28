/*
 * Vulkan Example - Fluid simulation
 *
 *
 * This code is licensed under the MIT license (MIT)
 * (http://opensource.org/licenses/MIT)
 */

#include <ktx.h>
#include <ktxvulkan.h>

#include "VulkanglTFModel.h"
#include "stb_image.h"
#include "vulkanexamplebase.h"

// Offscreen frame buffer properties
#define FB_COLOR_FORMAT VK_FORMAT_R32G32B32A32_SFLOAT

class VulkanExample : public VulkanExampleBase {
 public:
  // Enable Vulkan 1.3
  VkPhysicalDeviceVulkan13Features enabledFeatures13_{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};

  const uint32_t JACOBI_ITERATIONS = 100;
  // Inner slab offset (in pixels) for x and y axis
  const uint32_t SLAB_OFFSET = 0;
  static constexpr float TIME_STEP{1.f / 360};
  // 0 = color field
  // 1 = velocity field
  // 2 = pressure field
  static constexpr int WHICH_TEXTURE_DISPLAY_AT_START{0};
  static constexpr float IMPULSE_RADIUS{0.004f};
  bool showVelocityArrows_ = false;
  bool advectVelocity_ = true;
  std::vector<std::string> texture_viewer_selection = {"Color", "Velocity",
                                                       "Pressure"};

  struct Vertex {
    glm::vec2 pos;
    glm::vec2 translation;
    Vertex() = default;
    Vertex(glm::vec2 _p, glm::vec2 _t) : pos(_p), translation(_t) {}

    static VkVertexInputBindingDescription getBindingDescription() {
      VkVertexInputBindingDescription bindingDescription{};
      bindingDescription.binding = 0;
      bindingDescription.stride = sizeof(Vertex);
      bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

      return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2>
    getAttributeDescriptions() {
      std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
      attributeDescriptions[0].binding = 0;
      attributeDescriptions[0].location = 0;
      attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
      attributeDescriptions[0].offset = offsetof(Vertex, pos);

      attributeDescriptions[1].binding = 0;
      attributeDescriptions[1].location = 1;
      attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
      attributeDescriptions[1].offset = offsetof(Vertex, translation);

      return attributeDescriptions;
    }
  };
  std::vector<VulkanExample::Vertex> triangle_vertices_;

  struct ColorInitUBO {
    alignas(8) glm::vec2 bufferResolution{};
    alignas(4) int whichTexture{};
  };

  struct AdvectionUBO {
    alignas(8) glm::vec2 bufferResolution{};
    alignas(4) float timestep{TIME_STEP};
  };

  struct BoundaryUBO {
    alignas(8) glm::vec2 bufferResolution{};
    alignas(4) float scale{};
  };

  struct ImpulseUBO {
    alignas(8) glm::vec2 epicenter{};
    alignas(8) glm::vec2 bufferResolution{};
    alignas(8) glm::vec2 dxdy{};
    alignas(4) float radius{IMPULSE_RADIUS};
  };

  struct JacobiUBO {
    alignas(8) glm::vec2 bufferResolution{};
  };

  struct DivergenceUBO {
    alignas(8) glm::vec2 bufferResolution{};
    alignas(4) float timestep{TIME_STEP};
  };

  struct GradientUBO {
    alignas(8) glm::vec2 bufferResolution{};
    alignas(4) float timestep{TIME_STEP};
  };

  struct TextureViewSwitcherUBO {
    alignas(4) int chooseDisplayTexture{WHICH_TEXTURE_DISPLAY_AT_START};
  };

  struct VelocityArrowsUBO {};

  struct {
    ColorInitUBO colorInit;
    AdvectionUBO advection;
    ImpulseUBO impulse;
    BoundaryUBO boundary;
    JacobiUBO jacobi;
    DivergenceUBO divergence;
    GradientUBO gradient;
    TextureViewSwitcherUBO textureViewSwitcher;
    VelocityArrowsUBO velocityArrows;
  } ubos_;

  struct UniformBuffers {
    vks::Buffer colorInit;
    vks::Buffer velocityInit;
    vks::Buffer advection;
    vks::Buffer impulse;
    vks::Buffer boundary;
    vks::Buffer jacobi;
    vks::Buffer divergence;
    vks::Buffer gradient;
    vks::Buffer textureViewSwitcher;
    vks::Buffer velocityArrows;
  };
  std::array<UniformBuffers, MAX_CONCURRENT_FRAMES> uniformBuffers_{};

  vks::Buffer vertex_buffer_;

  struct {
    VkDescriptorSetLayout colorInit;
    VkDescriptorSetLayout advection;
    VkDescriptorSetLayout boundary;
    VkDescriptorSetLayout impulse;
    VkDescriptorSetLayout jacobi;
    VkDescriptorSetLayout divergence;
    VkDescriptorSetLayout gradient;
    VkDescriptorSetLayout textureViewSwitcher;
    VkDescriptorSetLayout velocityArrows;
    VkDescriptorSetLayout colorPass;
  } descriptorSetLayouts_{};

  struct DescriptorSets {
    VkDescriptorSet colorInit;
    VkDescriptorSet velocityInit;
    VkDescriptorSet advectColor;
    VkDescriptorSet advectVelocity;
    VkDescriptorSet impulse;
    VkDescriptorSet boundaryVelocity;
    VkDescriptorSet boundaryPressure;
    VkDescriptorSet boundaryColor;
    VkDescriptorSet jacobiPressure;
    VkDescriptorSet divergence;
    VkDescriptorSet gradient;
    VkDescriptorSet textureViewSwitcher;
    VkDescriptorSet velocityArrows;
    VkDescriptorSet colorPass;
  };
  std::array<DescriptorSets, MAX_CONCURRENT_FRAMES> descriptorSets_{};

  struct {
    VkPipelineLayout colorInit;
    VkPipelineLayout velocityInit;
    VkPipelineLayout advection;
    VkPipelineLayout impulse;
    VkPipelineLayout boundary;
    VkPipelineLayout jacobi;
    VkPipelineLayout divergence;
    VkPipelineLayout gradient;
    VkPipelineLayout textureViewSwitcher;
    VkPipelineLayout velocityArrows;
    VkPipelineLayout colorPass;
  } pipelineLayouts_{};

  struct {
    VkPipeline velocityInit;
    VkPipeline colorInit;
    VkPipeline advection;
    VkPipeline impulse;
    VkPipeline boundary;
    VkPipeline jacobi;
    VkPipeline divergence;
    VkPipeline gradient;
    VkPipeline textureViewSwitcher;
    VkPipeline velocityArrows;
    VkPipeline colorPass;
  } pipelines_{};

  // Structure to hold offscreen velocity buffer
  struct TextureFieldBuffer {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView imageView;
    VkDescriptorImageInfo descriptor;
    VkFormat format;
    VkExtent2D extent;
  };
  struct OffscreenPass {
    VkSampler sampler{};
  } offscreenPass_;

  // 2 framebuffers for each field, index 0 is for reading, 1 is for writing
  std::array<TextureFieldBuffer, 2> color_field_{};  // Scalar valued color map
  std::array<TextureFieldBuffer, 2> velocity_field_{};
  std::array<TextureFieldBuffer, 2> pressure_field_{};
  TextureFieldBuffer divergence_field_{};  // Scalar valued
  // texture view switcher FB needs to be offscreen to overlay velocity arrows
  std::array<TextureFieldBuffer, 2> color_pass_{};
  // feedback control for mouse click + movement
  bool addImpulse_ = false;
  bool shouldInitColorField_ = true;
  bool shouldInitVelocityField_ = true;

  // Debug Extension
  std::vector<float> debugColor = {.7f, 0.4f, 0.4f, 1.0f};
  PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT{nullptr};
  PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT{nullptr};

  VulkanExample() {
    title_ = "Fluid Simulation";
    camera_.type_ = Camera::CameraType::lookat;
    camera_.setPosition(glm::vec3(0.0f, 0.0f, -15.0f));
    camera_.setRotation(glm::vec3(0.0f));
    camera_.setRotationSpeed(0.25f);
    camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);

    apiVersion_ = VK_API_VERSION_1_3;

    // Dynamic rendering
    enabledFeatures13_.dynamicRendering = VK_TRUE;

    deviceCreatepNextChain_ = &enabledFeatures13_;
  }

  // (Part A)
  void prepare() override {
    VulkanExampleBase::prepare();
    prepareUniformBuffers();
    prepareVertices();
    prepareOffscreen();
    setupDescriptors();
    preparePipelines();
    prepareDebug();
    prepared_ = true;
  }

  void prepareVertices() {
    triangle_vertices_.clear();
    const std::array<glm::vec2, 3> triangle = {
        glm::vec2(0.0f, 0.2f), glm::vec2(1.0f, 0.f), glm::vec2(0.0f, -.2f)};

    int arrow_spacing = 30;
    for (int y = arrow_spacing / 2.f; y < height_; y += arrow_spacing) {
      for (int x = arrow_spacing / 2.f; x < width_; x += arrow_spacing) {
        for (int i = 0; i < 3; i++) {
          triangle_vertices_.emplace_back(
              triangle[i],
              glm::vec2(2.f * x / width_ - 1, 2.f * y / height_ - 1));
        }
      }
    }

    VkBufferCreateInfo bufferInfo{};
    VK_CHECK_RESULT(vulkanDevice_->createBuffer(
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &vertex_buffer_,
        triangle_vertices_.size() * sizeof(VulkanExample::Vertex),
        triangle_vertices_.data()));
    VK_CHECK_RESULT(vertex_buffer_.map());
  }

  void prepareUniformBuffers() {
    for (auto& buffer : uniformBuffers_) {
      // Color Init
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.colorInit, sizeof(ColorInitUBO), &ubos_.colorInit));
      VK_CHECK_RESULT(buffer.colorInit.map());

      // Velocity Init
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.velocityInit, sizeof(ColorInitUBO), &ubos_.colorInit));
      VK_CHECK_RESULT(buffer.velocityInit.map());

      // Advection
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.advection, sizeof(AdvectionUBO), &ubos_.advection));
      VK_CHECK_RESULT(buffer.advection.map());

      // Boundary
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.boundary, sizeof(BoundaryUBO), &ubos_.boundary));
      VK_CHECK_RESULT(buffer.boundary.map());

      // Jacobi
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.jacobi, sizeof(JacobiUBO), &ubos_.jacobi));
      VK_CHECK_RESULT(buffer.jacobi.map());

      // Divergence
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.divergence, sizeof(DivergenceUBO), &ubos_.divergence));
      VK_CHECK_RESULT(buffer.divergence.map());

      // Gradient
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.gradient, sizeof(GradientUBO), &ubos_.gradient));
      VK_CHECK_RESULT(buffer.gradient.map());

      // Impulse
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.impulse, sizeof(ImpulseUBO), &ubos_.impulse));
      VK_CHECK_RESULT(buffer.impulse.map());

      // texture view switcher
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.textureViewSwitcher, sizeof(TextureViewSwitcherUBO),
          &ubos_.textureViewSwitcher));
      VK_CHECK_RESULT(buffer.textureViewSwitcher.map());

      // Velocity arrows
      VK_CHECK_RESULT(vulkanDevice_->createBuffer(
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          &buffer.velocityArrows, sizeof(VelocityArrowsUBO),
          &ubos_.velocityArrows));
      VK_CHECK_RESULT(buffer.velocityArrows.map());
    }
  }

  void prepareOffscreen() {
    // Create sampler to sample from the color attachments
    VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.addressModeV = sampler.addressModeU;
    sampler.addressModeW = sampler.addressModeU;
    sampler.mipLodBias = 0.0f;
    sampler.maxAnisotropy = 1.0f;
    sampler.minLod = 0.0f;
    sampler.maxLod = 1.0f;
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(
        vkCreateSampler(device_, &sampler, nullptr, &offscreenPass_.sampler));

    // Color field
    for (auto& fb : color_field_) {
      prepareOffscreenFramebuffer(&fb, FB_COLOR_FORMAT);
    }

    // Velocity field
    for (auto& fb : velocity_field_) {
      prepareOffscreenFramebuffer(&fb, FB_COLOR_FORMAT);
    }

    // Pressure field
    for (auto& fb : pressure_field_) {
      prepareOffscreenFramebuffer(&fb, FB_COLOR_FORMAT);
    }

    // Divergence field
    prepareOffscreenFramebuffer(&divergence_field_, FB_COLOR_FORMAT);

    // texture view switcher
    for (auto& fb : color_pass_) {
      prepareOffscreenFramebuffer(&fb, FB_COLOR_FORMAT);
    }
  }

  void setupDescriptors() {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            /* descriptorCount */ 13 * MAX_CONCURRENT_FRAMES *
                /*max number of uniform buffers*/ 1),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            /* descriptorCount */ MAX_CONCURRENT_FRAMES *
                /*max number of textures*/ 18)};
    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(
            poolSizes,
            /* max number of descriptor sets that can be allocated at once*/
            15 * MAX_CONCURRENT_FRAMES);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo,
                                           nullptr, &descriptorPool_));
    // Layout: Color init
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        // Binding 0 : uniform buffer
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),
    };

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI =
        vks::initializers::descriptorSetLayoutCreateInfo(
            setLayoutBindings.data(),
            static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.colorInit));

    // Layout: Velocity Init
    setLayoutBindings = {
        // Binding 0 : Fragment shader
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),
    };
    descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.colorInit));

    // Layout: Advection
    setLayoutBindings = {
        // Binding 0 : Fragment shader
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),
        // Binding 1 : Fragment shader field texture 1
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 1),
        // Binding 2 : Fragment shader field texture 2
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 2),
    };

    descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.advection));

    // Layout: Impulse
    setLayoutBindings = {
        // Binding 0 : Fragment shader
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),
        // Binding 1 : Fragment shader field texture 1
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 1)};
    descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.impulse));
    // Layout: Boundary
    setLayoutBindings = {
        // Binding 0 : Fragment shader
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),
        // Binding 1 : Fragment shader field texture 1
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 1)};
    descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.boundary));

    // Layout: Jacobi
    setLayoutBindings = {
        // Binding 0 : Fragment shader
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),
        // Binding 1 : Fragment shader field texture 1
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 1),
        // Binding 2 : Fragment shader field texture 2
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 2),
    };
    descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI,
                                                nullptr,
                                                &descriptorSetLayouts_.jacobi));

    // Layout: Divergence
    setLayoutBindings = {
        // Binding 0 : Fragment shader
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),
        // Binding 1 : Fragment shader field texture 1
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 1),
    };
    descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.divergence));

    // Layout: Gradient
    setLayoutBindings = {
        // Binding 0 : Uniform
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),
        // Binding 1 : velocity field
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 1),
        // Binding 2 : pressure field
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 2),
    };
    descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.gradient));

    // Layout: Texture view switcher
    setLayoutBindings = {
        // Binding 0 : Uniform
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),
        // Binding 1 : Color field
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 1),
        // Binding 2 : Velocity field
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 2),
        // Binding 3 : Pressure field
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 3),
    };
    descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device_, &descriptorSetLayoutCI, nullptr,
        &descriptorSetLayouts_.textureViewSwitcher));

    // Layout: Velocity Arrows
    setLayoutBindings = {
        // Binding 0 : Uniform
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
            /*binding id*/ 0),
        // Binding 1 : Velocity field texture map
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_VERTEX_BIT,
            /*binding id*/ 1),
    };
    descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.velocityArrows));

    // Layout: Color pass
    setLayoutBindings = {
        // Binding 0 : final texture map to show to screen
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            /*binding id*/ 0),
    };
    descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCI, nullptr,
                                    &descriptorSetLayouts_.colorPass));

    // Descriptor Sets
    for (auto i = 0; i < uniformBuffers_.size(); i++) {
      // Color Init
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool_, &descriptorSetLayouts_.colorInit, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].colorInit));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].colorInit, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].colorInit.descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Velocity Init
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.colorInit, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &descriptorSets_[i].velocityInit));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].velocityInit,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].velocityInit.descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Advection: Color + Velocity
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.advection, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &descriptorSets_[i].advectColor));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].advectColor, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].advection.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].advectColor,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &velocity_field_[0].descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].advectColor,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 2, &color_field_[0].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Advection: Velocity + Velocity
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.advection, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &descriptorSets_[i].advectVelocity));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].advectVelocity,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].advection.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].advectVelocity,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &velocity_field_[0].descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].advectVelocity,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 2, &velocity_field_[0].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
      // Impulse
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.impulse, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].impulse));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].impulse, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].impulse.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].impulse,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &velocity_field_[0].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Boundary: Velocity
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.boundary, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &descriptorSets_[i].boundaryVelocity));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].boundaryVelocity,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].boundary.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].boundaryVelocity,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &velocity_field_[0].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Boundary: Pressure
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.boundary, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &descriptorSets_[i].boundaryPressure));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].boundaryPressure,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].boundary.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].boundaryPressure,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &pressure_field_[0].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Boundary: Color
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.boundary, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &descriptorSets_[i].boundaryColor));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].boundaryColor,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].boundary.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].boundaryColor,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &color_field_[0].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Jacobi: Pressure
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.jacobi, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &descriptorSets_[i].jacobiPressure));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].jacobiPressure,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].jacobi.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].jacobiPressure,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &pressure_field_[0].descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].jacobiPressure,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 2, &divergence_field_.descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Divergence
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.divergence, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].divergence));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].divergence, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].divergence.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].divergence,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &velocity_field_[0].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Gradient
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.gradient, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].gradient));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].gradient, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].gradient.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].gradient,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &velocity_field_[0].descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].gradient,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 2, &pressure_field_[0].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Texture view switcher
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.textureViewSwitcher, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &descriptorSets_[i].textureViewSwitcher));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].textureViewSwitcher,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0,
              &uniformBuffers_[i].textureViewSwitcher.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].textureViewSwitcher,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &color_field_[0].descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].textureViewSwitcher,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 2, &velocity_field_[0].descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].textureViewSwitcher,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 3, &pressure_field_[0].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Velocity arrows
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.velocityArrows, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device_, &allocInfo, &descriptorSets_[i].velocityArrows));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].velocityArrows,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              /*binding id*/ 0, &uniformBuffers_[i].velocityArrows.descriptor),
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].velocityArrows,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 1, &velocity_field_[0].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);

      // Color pass
      allocInfo = vks::initializers::descriptorSetAllocateInfo(
          descriptorPool_, &descriptorSetLayouts_.colorPass, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo,
                                               &descriptorSets_[i].colorPass));
      writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              descriptorSets_[i].colorPass,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              /*binding id*/ 0, &color_pass_[0].descriptor),
      };
      vkUpdateDescriptorSets(device_,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }
  }

  void preparePipelines() {
    VkPipelineLayoutCreateInfo pipelineLayoutCI =
        vks::initializers::pipelineLayoutCreateInfo(
            &descriptorSetLayouts_.colorInit, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.colorInit));

    pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.colorInit, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.velocityInit));

    pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.advection, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.advection));

    pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.impulse, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.impulse));

    pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.boundary, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.boundary));

    pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.jacobi, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.jacobi));

    pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.divergence, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.divergence));

    pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.gradient, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.gradient));

    pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.textureViewSwitcher, 1);
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                               &pipelineLayouts_.textureViewSwitcher));

    pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.velocityArrows, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.velocityArrows));

    pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(
        &descriptorSetLayouts_.colorPass, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr,
                                           &pipelineLayouts_.colorPass));

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
        vks::initializers::pipelineCreateInfo();
    pipelineCI.pInputAssemblyState = &inputAssemblyState;
    pipelineCI.pRasterizationState = &rasterizationState;
    pipelineCI.pColorBlendState = &colorBlendState;
    pipelineCI.pMultisampleState = &multisampleState;
    pipelineCI.pViewportState = &viewportState;
    pipelineCI.pDepthStencilState = &depthStencilState;
    pipelineCI.pDynamicState = &dynamicState;
    pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCI.pStages = shaderStages.data();

    // New create info to define color, depth and stencil attachments at
    // pipeline create time
    VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo{};
    pipelineRenderingCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    pipelineRenderingCreateInfo.colorAttachmentCount = 1;
    pipelineRenderingCreateInfo.depthAttachmentFormat = depthFormat_;
    pipelineRenderingCreateInfo.stencilAttachmentFormat = depthFormat_;
    // Chain into the pipeline create info
    pipelineCI.pNext = &pipelineRenderingCreateInfo;
    pipelineCI.renderPass = VK_NULL_HANDLE;  // No render pass!

    // Advection pipeline
    VkPipelineVertexInputStateCreateInfo emptyInputState =
        vks::initializers::pipelineVertexInputStateCreateInfo();
    pipelineCI.pVertexInputState = &emptyInputState;
    pipelineCI.layout = pipelineLayouts_.advection;
    shaderStages[0] = loadShader(getShadersPath() + "fluidsim/simple.vert.spv",
                                 VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] =
        loadShader(getShadersPath() + "fluidsim/advection.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &velocity_field_[1].format;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.advection));

    // Color init pipeline
    pipelineCI.layout = pipelineLayouts_.colorInit;
    shaderStages[1] =
        loadShader(getShadersPath() + "fluidsim/colorinit.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &color_field_[1].format;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.colorInit));

    // Velocity init pipeline
    pipelineCI.layout = pipelineLayouts_.velocityInit;
    shaderStages[1] =
        loadShader(getShadersPath() + "fluidsim/velocityinit.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &velocity_field_[1].format;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.velocityInit));

    // Boundary pipeline
    pipelineCI.layout = pipelineLayouts_.boundary;
    shaderStages[1] =
        loadShader(getShadersPath() + "fluidsim/boundary.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    // TODO: boundary needs its own vkformat?
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &velocity_field_[1].format;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.boundary));

    // Impulse Pipeline
    pipelineCI.layout = pipelineLayouts_.impulse;
    shaderStages[1] = loadShader(getShadersPath() + "fluidsim/impulse.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &velocity_field_[1].format;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.impulse));

    // Jacobi pipeline
    pipelineCI.layout = pipelineLayouts_.jacobi;
    shaderStages[1] = loadShader(getShadersPath() + "fluidsim/jacobi.frag.spv",
                                 VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &pressure_field_[1].format;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.jacobi));

    // Divergence pipeline
    pipelineCI.layout = pipelineLayouts_.divergence;
    shaderStages[1] =
        loadShader(getShadersPath() + "fluidsim/divergence.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &divergence_field_.format;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.divergence));

    // Gradient pipeline
    pipelineCI.layout = pipelineLayouts_.gradient;
    shaderStages[1] =
        loadShader(getShadersPath() + "fluidsim/gradient.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &velocity_field_[1].format;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.gradient));

    // Texture view switcher pipeline
    pipelineCI.layout = pipelineLayouts_.textureViewSwitcher;
    shaderStages[1] =
        loadShader(getShadersPath() + "fluidsim/textureviewswitcher.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &color_pass_[1].format;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.textureViewSwitcher));

    // Color pass pipeline
    pipelineCI.layout = pipelineLayouts_.colorPass;
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &swapChain_.colorFormat_;
    shaderStages[1] =
        loadShader(getShadersPath() + "fluidsim/colorpass.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.colorPass));

    // Arrow vector pipeline
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescription = Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputCI =
        vks::initializers::pipelineVertexInputStateCreateInfo();
    vertexInputCI.vertexBindingDescriptionCount = 1;
    vertexInputCI.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescription.size());
    vertexInputCI.pVertexBindingDescriptions = &bindingDescription;
    vertexInputCI.pVertexAttributeDescriptions = attributeDescription.data();

    pipelineCI.pVertexInputState = &vertexInputCI;
    shaderStages[0] =
        loadShader(getShadersPath() + "fluidsim/velocityarrows.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT);
    pipelineCI.layout = pipelineLayouts_.velocityArrows;
    shaderStages[1] =
        loadShader(getShadersPath() + "fluidsim/velocityarrows.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1,
                                              &pipelineCI, nullptr,
                                              &pipelines_.velocityArrows));
  }

  void prepareDebug() {
    vkCmdBeginDebugUtilsLabelEXT =
        reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
            vkGetInstanceProcAddr(instance_, "vkCmdBeginDebugUtilsLabelEXT"));
    vkCmdEndDebugUtilsLabelEXT =
        reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
            vkGetInstanceProcAddr(instance_, "vkCmdEndDebugUtilsLabelEXT"));
  }

  void cmdBeginLabel(const VkCommandBuffer command_buffer,
                     const char* label_name,
                     std::vector<float> color) {
    VkDebugUtilsLabelEXT label = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = label_name;
    memcpy(label.color, color.data(), sizeof(float) * 4);
    vkCmdBeginDebugUtilsLabelEXT(command_buffer, &label);
  }

  void cmdEndLabel(VkCommandBuffer command_buffer) {
    vkCmdEndDebugUtilsLabelEXT(command_buffer);
  }

  // Part B (rendering)
  void render() override {
    if (!prepared_) {
      return;
    }
    VulkanExampleBase::prepareFrame();
    updateUniformBuffers();
    buildCommandBuffer();
    VulkanExampleBase::submitFrame();
  }

  // B.1
  void updateUniformBuffers() {
    ubos_.colorInit.bufferResolution = glm::vec2(width_, height_);
    memcpy(uniformBuffers_[currentBuffer_].colorInit.mapped, &ubos_.colorInit,
           sizeof(ColorInitUBO));

    memcpy(uniformBuffers_[currentBuffer_].velocityInit.mapped,
           &ubos_.colorInit, sizeof(ColorInitUBO));

    ubos_.advection.bufferResolution = glm::vec2(width_, height_);
    memcpy(uniformBuffers_[currentBuffer_].advection.mapped, &ubos_.advection,
           sizeof(AdvectionUBO));

    ubos_.boundary.bufferResolution = glm::vec2(width_, height_);
    memcpy(uniformBuffers_[currentBuffer_].boundary.mapped, &ubos_.boundary,
           sizeof(BoundaryUBO));

    ubos_.jacobi.bufferResolution = glm::vec2(width_, height_);
    memcpy(uniformBuffers_[currentBuffer_].jacobi.mapped, &ubos_.jacobi,
           sizeof(JacobiUBO));

    ubos_.gradient.bufferResolution = glm::vec2(width_, height_);
    memcpy(uniformBuffers_[currentBuffer_].gradient.mapped, &ubos_.gradient,
           sizeof(GradientUBO));

    ubos_.divergence.bufferResolution = glm::vec2(width_, height_);
    memcpy(uniformBuffers_[currentBuffer_].divergence.mapped, &ubos_.divergence,
           sizeof(DivergenceUBO));

    ubos_.impulse.bufferResolution = glm::vec2(width_, height_);
    memcpy(uniformBuffers_[currentBuffer_].impulse.mapped, &ubos_.impulse,
           sizeof(ImpulseUBO));

    memcpy(uniformBuffers_[currentBuffer_].textureViewSwitcher.mapped,
           &ubos_.textureViewSwitcher, sizeof(TextureViewSwitcherUBO));

    memcpy(uniformBuffers_[currentBuffer_].velocityArrows.mapped,
           &ubos_.velocityArrows, sizeof(VelocityArrowsUBO));
  }

  void OnUpdateUIOverlay(vks::UIOverlay* overlay) override {
    if (overlay->header("Settings")) {
      overlay->comboBox("Select Texture Map",
                        &ubos_.textureViewSwitcher.chooseDisplayTexture,
                        texture_viewer_selection);
      overlay->checkBox("Show velocity arrows", &showVelocityArrows_);
      if (overlay->checkBox("Advect velocity", &advectVelocity_)) {
        windowResized();
      }
      if (overlay->button("Reset")) {
        windowResized();
      }
    }
  }

  // B.2
  void buildCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    if (shouldInitColorField_) {
      ubos_.colorInit.whichTexture = 0;
      memcpy(uniformBuffers_[currentBuffer_].colorInit.mapped, &ubos_.colorInit,
             sizeof(BoundaryUBO));
      initColorCmd(cmdBuffer);
      copyImageCmd(cmdBuffer, color_field_);
      shouldInitColorField_ = false;
    }

    if (shouldInitVelocityField_) {
      initVelocityCmd(cmdBuffer);
      copyImageCmd(cmdBuffer, velocity_field_);
      shouldInitVelocityField_ = false;
    }

    // Advect Velocity
    if (advectVelocity_) {
      advectVelocityCmd(cmdBuffer);
      copyImageCmd(cmdBuffer, velocity_field_);
    }

    // Impulse
    if (addImpulse_) {
      impulseCmd(cmdBuffer);
      copyImageCmd(cmdBuffer, velocity_field_);
      addImpulse_ = false;
    }

    // Divergence
    divergenceCmd(cmdBuffer);

    // Jacobi Iteration: Pressure
    cmdBeginLabel(cmdBuffer, "Jacobi for Pressure", debugColor);
    for (uint32_t i = 0; i < JACOBI_ITERATIONS; i++) {
      pressureJacobiCmd(cmdBuffer);
      copyImageCmd(cmdBuffer, pressure_field_);
    }
    cmdEndLabel(cmdBuffer);

    // Gradient subtraction
    gradientSubtractionCmd(cmdBuffer);
    copyImageCmd(cmdBuffer, velocity_field_);

    // Advect Color
    advectColorCmd(cmdBuffer);
    copyImageCmd(cmdBuffer, color_field_);

    // Select which tex to view
    textureViewSwitcherCmd(cmdBuffer);
    copyImageCmd(cmdBuffer, color_pass_);

    if (showVelocityArrows_) {
      // Draw arrows
      velocityArrowsCmd(cmdBuffer);
      copyImageCmd(cmdBuffer, color_pass_);
    }

    // Color pass
    colorPassCmd(cmdBuffer);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void initColorCmd(const VkCommandBuffer& cmdBuffer) {
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, color_field_[1].image, 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    VkRenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachment.imageView = color_field_[1].imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {0.0f, 0.0f, 0.0f, 0.0f};

    VkRenderingInfoKHR renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    renderingInfo.renderArea = {int(SLAB_OFFSET), int(SLAB_OFFSET),
                                width_ - 2 * SLAB_OFFSET,
                                height_ - 2 * SLAB_OFFSET};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = nullptr;
    renderingInfo.pStencilAttachment = nullptr;

    cmdBeginLabel(cmdBuffer, "Initialize Color Field", debugColor);
    vkCmdBeginRendering(cmdBuffer, &renderingInfo);

    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.colorInit,
        0, 1, &descriptorSets_[currentBuffer_].colorInit, 0, nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.colorInit);
    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

    vkCmdEndRendering(cmdBuffer);
    cmdEndLabel(cmdBuffer);
  }

  void initVelocityCmd(const VkCommandBuffer& cmdBuffer) {
    // Transition image to correct layout first
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, velocity_field_[1].image, 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    VkRenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachment.imageView = velocity_field_[1].imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {0.0f, 0.0f, 0.0f, 0.0f};

    VkRenderingInfoKHR renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    renderingInfo.renderArea = {int(SLAB_OFFSET), int(SLAB_OFFSET),
                                width_ - 2 * SLAB_OFFSET,
                                height_ - 2 * SLAB_OFFSET};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    cmdBeginLabel(cmdBuffer, "Initialize Velocity Field", debugColor);
    vkCmdBeginRendering(cmdBuffer, &renderingInfo);

    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayouts_.velocityInit, 0, 1,
                            &descriptorSets_[currentBuffer_].velocityInit, 0,
                            nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.velocityInit);
    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

    vkCmdEndRendering(cmdBuffer);
    cmdEndLabel(cmdBuffer);
  }

  void advectColorCmd(const VkCommandBuffer& cmdBuffer) {
    cmdBeginLabel(cmdBuffer, "Advecting Color", debugColor);
    advectionCmd(cmdBuffer, color_field_,
                 &descriptorSets_[currentBuffer_].advectColor);
    cmdEndLabel(cmdBuffer);
  }

  void advectVelocityCmd(const VkCommandBuffer& cmdBuffer) {
    cmdBeginLabel(cmdBuffer, "Advecting velocity", debugColor);
    advectionCmd(cmdBuffer, velocity_field_,
                 &descriptorSets_[currentBuffer_].advectVelocity);
    cmdEndLabel(cmdBuffer);
  }

  void advectionCmd(const VkCommandBuffer& cmdBuffer,
                    const std::array<TextureFieldBuffer, 2>& output_field,
                    const VkDescriptorSet* descriptor_set) const {
    // Transition image to correct layout first
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, output_field[1].image, 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    VkRenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachment.imageView = output_field[1].imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {0.0f, 0.0f, 0.0f, 0.0f};

    VkRenderingInfoKHR renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    renderingInfo.renderArea = {int(SLAB_OFFSET), int(SLAB_OFFSET),
                                width_ - 2 * SLAB_OFFSET,
                                height_ - 2 * SLAB_OFFSET};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(cmdBuffer, &renderingInfo);

    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayouts_.advection, 0, 1, descriptor_set, 0,
                            nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.advection);
    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

    vkCmdEndRendering(cmdBuffer);
  }

  void impulseCmd(const VkCommandBuffer& cmdBuffer) {
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, velocity_field_[1].image, 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    VkRenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachment.imageView = velocity_field_[1].imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {0.0f, 0.0f, 0.0f, 0.0f};

    VkRenderingInfoKHR renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    renderingInfo.renderArea = {int(SLAB_OFFSET), int(SLAB_OFFSET),
                                width_ - 2 * SLAB_OFFSET,
                                height_ - 2 * SLAB_OFFSET};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    cmdBeginLabel(cmdBuffer, "Adding impulse", debugColor);
    vkCmdBeginRendering(cmdBuffer, &renderingInfo);

    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.impulse, 0,
        1, &descriptorSets_[currentBuffer_].impulse, 0, nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.impulse);
    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

    // IMPORTANT: This barrier is to serialize WRITES BEFORE READS
    {
      VkMemoryBarrier memBarrier = {};
      memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_HOST_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0,
                           nullptr, 0, nullptr);
    }
    vkCmdEndRendering(cmdBuffer);
    cmdEndLabel(cmdBuffer);
  }

  void velocityBoundaryCmd(const VkCommandBuffer& cmdBuffer) {
    ubos_.boundary.scale = -1;
    memcpy(uniformBuffers_[currentBuffer_].boundary.mapped, &ubos_.boundary,
           sizeof(BoundaryUBO));
    boundaryCmd(cmdBuffer, velocity_field_,
                &descriptorSets_[currentBuffer_].boundaryVelocity);
  }

  void pressureBoundaryCmd(VkCommandBuffer& cmdBuffer) {
    ubos_.boundary.scale = 1;
    memcpy(uniformBuffers_[currentBuffer_].boundary.mapped, &ubos_.boundary,
           sizeof(BoundaryUBO));
    boundaryCmd(cmdBuffer, pressure_field_,
                &descriptorSets_[currentBuffer_].boundaryPressure);
  }

  void colorBoundaryCmd(const VkCommandBuffer& cmdBuffer) {
    ubos_.boundary.scale = 1;
    memcpy(uniformBuffers_[currentBuffer_].boundary.mapped, &ubos_.boundary,
           sizeof(BoundaryUBO));
    boundaryCmd(cmdBuffer, color_field_,
                &descriptorSets_[currentBuffer_].boundaryColor);
  }

  void boundaryCmd(const VkCommandBuffer& cmdBuffer,
                   std::array<TextureFieldBuffer, 2>& output_field,
                   VkDescriptorSet* descriptor_set) {
    /*
    VkClearValue clearValues{};
    clearValues.color = {0.0f, 0.0f, 0.0f, 0.f};

    VkRenderPassBeginInfo renderPassBeginInfo =
        vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = offscreenPass_.renderPass;
    renderPassBeginInfo.framebuffer = output_field[1].framebuffer;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = width_;
    renderPassBeginInfo.renderArea.extent.height = height_;
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = &clearValues;

    vkCmdBeginRendering(cmdBuffer, &renderPassBeginInfo,
                         VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayouts_.boundary, 0, 1, descriptor_set, 0,
                            nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.boundary);
    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

    // IMPORTANT: This barrier is to serialize WRITES BEFORE READS
    {
      VkMemoryBarrier memBarrier = {};
      memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_HOST_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0,
                           nullptr, 0, nullptr);
    }
    vkCmdEndRendering(cmdBuffer);
*/
  }

  void pressureJacobiCmd(const VkCommandBuffer& cmdBuffer) {
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, pressure_field_[1].image, 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    VkRenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachment.imageView = pressure_field_[1].imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {0.0f, 0.0f, 0.0f, 0.0f};

    VkRenderingInfoKHR renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    renderingInfo.renderArea = {int(SLAB_OFFSET), int(SLAB_OFFSET),
                                width_ - 2 * SLAB_OFFSET,
                                height_ - 2 * SLAB_OFFSET};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(cmdBuffer, &renderingInfo);

    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.jacobi, 0,
        1, &descriptorSets_[currentBuffer_].jacobiPressure, 0, nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.jacobi);
    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

    // IMPORTANT: This barrier is to serialize WRITES BEFORE READS
    {
      VkMemoryBarrier memBarrier = {};
      memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_HOST_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0,
                           nullptr, 0, nullptr);
    }
    vkCmdEndRendering(cmdBuffer);
  }

  void divergenceCmd(const VkCommandBuffer& cmdBuffer) {
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, divergence_field_.image, 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    VkRenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachment.imageView = divergence_field_.imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {0.0f, 0.0f, 0.0f, 0.0f};

    VkRenderingInfoKHR renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    renderingInfo.renderArea = {int(SLAB_OFFSET), int(SLAB_OFFSET),
                                width_ - 2 * SLAB_OFFSET,
                                height_ - 2 * SLAB_OFFSET};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    cmdBeginLabel(cmdBuffer, "Divergence", debugColor);
    vkCmdBeginRendering(cmdBuffer, &renderingInfo);

    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.divergence,
        0, 1, &descriptorSets_[currentBuffer_].divergence, 0, nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.divergence);
    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

    // IMPORTANT: This barrier is to serialize WRITES BEFORE READS
    {
      VkMemoryBarrier memBarrier = {};
      memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_HOST_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0,
                           nullptr, 0, nullptr);
    }
    vkCmdEndRendering(cmdBuffer);
    cmdEndLabel(cmdBuffer);
  }

  void gradientSubtractionCmd(const VkCommandBuffer& cmdBuffer) {
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, velocity_field_[1].image, 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    VkRenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachment.imageView = velocity_field_[1].imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {0.0f, 0.0f, 0.0f, 0.0f};

    VkRenderingInfoKHR renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    renderingInfo.renderArea = {int(SLAB_OFFSET), int(SLAB_OFFSET),
                                width_ - 2 * SLAB_OFFSET,
                                height_ - 2 * SLAB_OFFSET};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    cmdBeginLabel(cmdBuffer, "Gradient Subtraction", debugColor);
    vkCmdBeginRendering(cmdBuffer, &renderingInfo);
    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.gradient,
        0, 1, &descriptorSets_[currentBuffer_].gradient, 0, nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.gradient);
    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

    // IMPORTANT: This barrier is to serialize WRITES BEFORE READS
    {
      VkMemoryBarrier memBarrier = {};
      memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_HOST_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0,
                           nullptr, 0, nullptr);
    }
    vkCmdEndRendering(cmdBuffer);
    cmdEndLabel(cmdBuffer);
  }

  void textureViewSwitcherCmd(const VkCommandBuffer& cmdBuffer) {
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, color_pass_[1].image, 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    VkRenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachment.imageView = color_pass_[1].imageView;
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

    cmdBeginLabel(cmdBuffer, "Texture View Switcher", debugColor);
    vkCmdBeginRendering(cmdBuffer, &renderingInfo);

    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipelineLayouts_.textureViewSwitcher, 0, 1,
        &descriptorSets_[currentBuffer_].textureViewSwitcher, 0, nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.textureViewSwitcher);
    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);

    // IMPORTANT: This barrier is to serialize WRITES BEFORE READS
    {
      VkMemoryBarrier memBarrier = {};
      memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_HOST_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0,
                           nullptr, 0, nullptr);
    }
    vkCmdEndRendering(cmdBuffer);
    cmdEndLabel(cmdBuffer);
  }

  void velocityArrowsCmd(const VkCommandBuffer& cmdBuffer) {
    vks::tools::insertImageMemoryBarrier(
        cmdBuffer, color_pass_[1].image, 0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    VkRenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachment.imageView = color_pass_[1].imageView;
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

    cmdBeginLabel(cmdBuffer, "Velocity Field Arrows", debugColor);
    vkCmdBeginRendering(cmdBuffer, &renderingInfo);
    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayouts_.velocityArrows, 0, 1,
                            &descriptorSets_[currentBuffer_].velocityArrows, 0,
                            nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.velocityArrows);
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &vertex_buffer_.buffer, offsets);

    vkCmdDraw(cmdBuffer, static_cast<uint32_t>(triangle_vertices_.size()), 1, 0,
              0);
    // IMPORTANT: This barrier is to serialize WRITES BEFORE READS
    {
      VkMemoryBarrier memBarrier = {};
      memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_HOST_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0,
                           nullptr, 0, nullptr);
    }
    vkCmdEndRendering(cmdBuffer);
    cmdEndLabel(cmdBuffer);
  }

  void colorPassCmd(const VkCommandBuffer& cmdBuffer) {
    // New structures are used to define the attachments used in dynamic
    // rendering
    VkRenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachment.imageView = swapChain_.imageViews_[currentImageIndex_];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {0, 1, 0};

    VkRenderingInfoKHR renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    renderingInfo.renderArea = {0, 0, width_, height_};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = nullptr;
    renderingInfo.pStencilAttachment = nullptr;

    cmdBeginLabel(cmdBuffer, "Color Pass", debugColor);
    vkCmdBeginRendering(cmdBuffer, &renderingInfo);
    VkViewport viewport =
        vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.colorPass,
        0, 1, &descriptorSets_[currentBuffer_].colorPass, 0, nullptr);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipelines_.colorPass);
    vkCmdDraw(cmdBuffer, 6, 1, 0, 0);
    drawUI(cmdBuffer);

    // IMPORTANT: This barrier is to serialize WRITES BEFORE READS
    {
      VkMemoryBarrier memBarrier = {};
      memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_HOST_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0,
                           nullptr, 0, nullptr);
    }
    vkCmdEndRendering(cmdBuffer);
    cmdEndLabel(cmdBuffer);
  }

  // Copy framebuffer color attachment from source to dest
  void copyImageCmd(const VkCommandBuffer& cmdBuffer,
                    const std::array<TextureFieldBuffer, 2>& framebuffers) {
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

    copyRegion.extent.width = static_cast<uint32_t>(width_);
    copyRegion.extent.height = static_cast<uint32_t>(height_);
    copyRegion.extent.depth = 1;

    cmdBeginLabel(cmdBuffer, "Copying image", {1.0f, 0.78f, 0.05f, 1.0f});
    // Copy output of write to read buffer
    vkCmdCopyImage(cmdBuffer, framebuffers[1].image,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, framebuffers[0].image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
    cmdEndLabel(cmdBuffer);
  }

  void windowResized() override {
    destroyVertexBuffer();
    destroyOffscreenPass();
    prepareOffscreen();
    vkResetDescriptorPool(device_, descriptorPool_, 0);
    setupDescriptors();
    prepareVertices();

    shouldInitColorField_ = true;
    shouldInitVelocityField_ = true;
    resized_ = false;
  }

  void destroyVertexBuffer() const {
    vkDestroyBuffer(device_, vertex_buffer_.buffer, nullptr);
  }

  void destroyOffscreenPass() {}

  void prepareOffscreenFramebuffer(TextureFieldBuffer* frameBuf,
                                   const VkFormat colorFormat) const {
    frameBuf->format = colorFormat;

    // Color attachment
    VkImageCreateInfo imageCI = vks::initializers::imageCreateInfo();
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = colorFormat;
    imageCI.extent.width = width_;
    imageCI.extent.height = height_;
    imageCI.extent.depth = 1;
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // We will sample directly from the color attachment
    imageCI.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VK_CHECK_RESULT(
        vkCreateImage(device_, &imageCI, nullptr, &frameBuf->image));

    VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
    VkMemoryRequirements memReqs;

    vkGetImageMemoryRequirements(device_, frameBuf->image, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = vulkanDevice_->getMemoryType(
        memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(
        vkAllocateMemory(device_, &memAlloc, nullptr, &frameBuf->memory));
    VK_CHECK_RESULT(
        vkBindImageMemory(device_, frameBuf->image, frameBuf->memory, 0));

    VkImageViewCreateInfo colorImageView =
        vks::initializers::imageViewCreateInfo();
    colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
    colorImageView.format = colorFormat;
    colorImageView.image = frameBuf->image;
    colorImageView.flags = 0;
    colorImageView.subresourceRange = {};
    colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorImageView.subresourceRange.baseMipLevel = 0;
    colorImageView.subresourceRange.levelCount = 1;
    colorImageView.subresourceRange.baseArrayLayer = 0;
    colorImageView.subresourceRange.layerCount = 1;

    VK_CHECK_RESULT(vkCreateImageView(device_, &colorImageView, nullptr,
                                      &frameBuf->imageView));

    // Fill a descriptor for later use in a descriptor set
    frameBuf->descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    frameBuf->descriptor.imageView = frameBuf->imageView;
    frameBuf->descriptor.sampler = offscreenPass_.sampler;
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

  void mouseMoved(double x, double y, bool& handled) override {
    if (mouseState.buttons.left) {
      float dx = mouseState.position.x - x;
      float dy = mouseState.position.y - y;
      ubos_.impulse.dxdy = glm::vec2(dx, dy);
      ubos_.impulse.epicenter = glm::vec2((float)x, (float)y);
      handled = true;
      addImpulse_ = true;
      return;
    }
  }

  ~VulkanExample() {
    if (device_) {
      vkDestroyPipeline(device_, pipelines_.colorInit, nullptr);
      vkDestroyPipeline(device_, pipelines_.advection, nullptr);
      vkDestroyPipeline(device_, pipelines_.boundary, nullptr);
      vkDestroyPipeline(device_, pipelines_.divergence, nullptr);
      vkDestroyPipeline(device_, pipelines_.jacobi, nullptr);
      vkDestroyPipeline(device_, pipelines_.textureViewSwitcher, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.colorInit, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.advection, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.boundary, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.jacobi, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.divergence, nullptr);
      vkDestroyPipelineLayout(device_, pipelineLayouts_.textureViewSwitcher,
                              nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.colorInit,
                                   nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.advection,
                                   nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.boundary,
                                   nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.divergence,
                                   nullptr);
      vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.jacobi,
                                   nullptr);
      vkDestroyDescriptorSetLayout(
          device_, descriptorSetLayouts_.textureViewSwitcher, nullptr);
      for (auto& buffer : uniformBuffers_) {
        buffer.colorInit.destroy();
        buffer.advection.destroy();
        buffer.boundary.destroy();
        buffer.divergence.destroy();
        buffer.jacobi.destroy();
      }
    }
  }
};

VULKAN_EXAMPLE_MAIN()
