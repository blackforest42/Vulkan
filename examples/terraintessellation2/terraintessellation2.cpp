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
  struct {
    vks::TextureCubeMap skyBox{};
    vks::Texture2D heightMap{};
  } textures_;

  VulkanExample() : VulkanExampleBase() {
    title = "Dynamic terrain tessellation";
    camera_.type_ = Camera::CameraType::firstperson;
    camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 512.0f);
    camera_.setRotation(glm::vec3(-12.0f, 159.0f, 0.0f));
    camera_.setTranslation(glm::vec3(18.0f, 22.5f, 57.5f));
    camera_.movementSpeed = 10.0f;
  }

  // Generate a terrain quad patch with normals based on heightmap data
  void generateTerrain() {
    std::vector<float> vertices;
    // Normalize to [0, 1] then rescale to 64
    float height_scale = 64.0f / 256.0f;
    // Adjusts vertical translation of height map. e.g. Below or above surface.
    float height_shift = 16.f;
    uint32_t ROWS = textures_.heightMap.height;
    uint32_t COLS = textures_.heightMap.width;

    for (int r = 0; r < ROWS; r++) {
      for (int c = 0; c < COLS; c++) {
      }
    }
  }

  void setupDescriptors() {}

  void preparePipelines() {}

  // Prepare and initialize uniform buffer containing shader uniforms
  void prepareUniformBuffers() {}

  void updateUniformBuffers() {}

  void prepare() {
    VulkanExampleBase::prepare();
    loadAssets();
    generateTerrain();
    prepareUniformBuffers();
    setupDescriptors();
    preparePipelines();
    prepared_ = true;
  }

  void buildCommandBuffer() {}

  void render() override {
    // if (!prepared_) {
    //   return;
    // }
    // VulkanExampleBase::prepareFrame();
    //  updateUniformBuffers();
    // buildCommandBuffer();
    // VulkanExampleBase::submitFrame();
  }

  virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay) {}

  // Enable physical device features required for this example
  virtual void getEnabledFeatures() {
    // Tessellation shader support is required for this example
    if (deviceFeatures_.tessellationShader) {
      enabledFeatures_.tessellationShader = VK_TRUE;
    } else {
      vks::tools::exitFatal(
          "Selected GPU does not support tessellation shaders!",
          VK_ERROR_FEATURE_NOT_PRESENT);
    }
    // Fill mode non solid is required for wireframe display
    if (deviceFeatures_.fillModeNonSolid) {
      enabledFeatures_.fillModeNonSolid = VK_TRUE;
    };
  }

  void loadAssets() {
    textures_.skyBox.loadFromFile(
        getAssetPath() + "textures/cartoon_sky_cubemap.ktx",
        VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice_, queue_);

    // Height data is stored in a one-channel texture
    textures_.heightMap.loadFromFile(
        getAssetPath() + "textures/iceland_heightmap.ktx", VK_FORMAT_R16_UNORM,
        vulkanDevice_, queue_);

    VkSamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();

    // Setup a mirroring sampler for the height map
    vkDestroySampler(device_, textures_.heightMap.sampler, nullptr);
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    samplerInfo.addressModeV = samplerInfo.addressModeU;
    samplerInfo.addressModeW = samplerInfo.addressModeU;
    samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = (float)textures_.heightMap.mipLevels;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(vkCreateSampler(device_, &samplerInfo, nullptr,
                                    &textures_.heightMap.sampler));
    textures_.heightMap.descriptor.sampler = textures_.heightMap.sampler;
  }

  ~VulkanExample() {
    if (device_) {
      textures_.heightMap.destroy();
      textures_.skyBox.destroy();
    }
  }
};

VULKAN_EXAMPLE_MAIN()
