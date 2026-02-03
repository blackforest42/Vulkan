/*
* Vulkan Example - Fullscreen radial blur (Single pass offscreen effect)
*
* This samples shows how to implement a simple post-processing effect
* 
* Copyright (C) 2016-2025 Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"

class VulkanExample : public VulkanExampleBase
{
public:
	bool blur = true;
	bool displayTexture = false;

	vks::Texture2D gradientTexture;
	vkglTF::Model scene;

	struct UniformDataScene {
		glm::mat4 projection;
		glm::mat4 modelView;
		float gradientPos = 0.0f;
	} uniformDataScene;

	struct UniformDataBlurParams {
		float radialBlurScale = 0.35f;
		float radialBlurStrength = 0.75f;
		glm::vec2 radialOrigin = glm::vec2(0.5f, 0.5f);
	} uniformDataBlurParams;

	struct UniformBuffers {
		vks::Buffer scene;
		vks::Buffer blurParams;
	};
	std::array<UniformBuffers, maxConcurrentFrames> uniformBuffers_;

	struct {
		VkPipeline radialBlur{ VK_NULL_HANDLE };
		VkPipeline colorPass{ VK_NULL_HANDLE };
		VkPipeline phongPass{ VK_NULL_HANDLE };
		VkPipeline offscreenDisplay{ VK_NULL_HANDLE };
	} pipelines_;

	struct {
		VkPipelineLayout radialBlur{ VK_NULL_HANDLE };
		VkPipelineLayout scene{ VK_NULL_HANDLE };
	} pipelineLayouts_;

	struct DescriptorSets {
		VkDescriptorSet scene{ VK_NULL_HANDLE };
		VkDescriptorSet radialBlur{ VK_NULL_HANDLE };
	};
	std::array<DescriptorSets, maxConcurrentFrames> descriptorSets_;

	struct {
		VkDescriptorSetLayout scene{ VK_NULL_HANDLE };
		VkDescriptorSetLayout radialBlur{ VK_NULL_HANDLE };
	} descriptorSetLayouts_;

	// Framebuffer for offscreen rendering
	struct FrameBufferAttachment {
		VkImage image;
		VkDeviceMemory mem;
		VkImageView view;
	};
	struct OffscreenPass {
		int32_t width, height;
		VkFramebuffer frameBuffer;
		FrameBufferAttachment color, depth;
		VkRenderPass renderPass;
		VkSampler sampler;
		VkDescriptorImageInfo descriptor;
	} offscreenPass_{};

	// Size of the shadow map texture (per face)
	const uint32_t offscreenImageSize{ 512 };
	// We use an 8 bit per component RGBA offscreen image for storing the scene parts that will be blurred
	const VkFormat offscreenImageFormat{ VK_FORMAT_R8G8B8A8_UNORM };

	VulkanExample() : VulkanExampleBase()
	{
		title = "Full screen radial blur effect";
		camera.type_ = Camera::CameraType::lookat;
		camera.setPosition(glm::vec3(0.0f, 0.0f, -17.5f));
		camera.setRotation(glm::vec3(-16.25f, -28.75f, 0.0f));
		camera.setPerspective(45.0f, (float)width / (float)height, 1.0f, 256.0f);
		timerSpeed *= 0.5f;
	}

	~VulkanExample()
	{
		if (device) {
			vkDestroyImageView(device, offscreenPass_.color.view, nullptr);
			vkDestroyImage(device, offscreenPass_.color.image, nullptr);
			vkFreeMemory(device, offscreenPass_.color.mem, nullptr);
			vkDestroyImageView(device, offscreenPass_.depth.view, nullptr);
			vkDestroyImage(device, offscreenPass_.depth.image, nullptr);
			vkFreeMemory(device, offscreenPass_.depth.mem, nullptr);
			vkDestroyRenderPass(device, offscreenPass_.renderPass, nullptr);
			vkDestroySampler(device, offscreenPass_.sampler, nullptr);
			vkDestroyFramebuffer(device, offscreenPass_.frameBuffer, nullptr);
			vkDestroyPipeline(device, pipelines_.radialBlur, nullptr);
			vkDestroyPipeline(device, pipelines_.phongPass, nullptr);
			vkDestroyPipeline(device, pipelines_.colorPass, nullptr);
			vkDestroyPipeline(device, pipelines_.offscreenDisplay, nullptr);
			vkDestroyPipelineLayout(device, pipelineLayouts_.radialBlur, nullptr);
			vkDestroyPipelineLayout(device, pipelineLayouts_.scene, nullptr);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayouts_.scene, nullptr);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayouts_.radialBlur, nullptr);
			for (auto& buffer : uniformBuffers_) {
				buffer.blurParams.destroy();
				buffer.scene.destroy();
			}
			gradientTexture.destroy();
		}
	}

	// Setup the offscreen framebuffer for rendering the blurred scene
	// The color attachment of this framebuffer will then be used to sample frame in the fragment shader of the final pass
	void prepareOffscreen()
	{
		offscreenPass_.width = offscreenImageSize;
		offscreenPass_.height = offscreenImageSize;

		// Find a suitable depth format
		VkFormat fbDepthFormat;
		VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &fbDepthFormat);
		assert(validDepthFormat);

		// Color attachment
		VkImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = VK_IMAGE_TYPE_2D;
		image.format = offscreenImageFormat;
		image.extent.width = offscreenPass_.width;
		image.extent.height = offscreenPass_.height;
		image.extent.depth = 1;
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = VK_SAMPLE_COUNT_1_BIT;
		image.tiling = VK_IMAGE_TILING_OPTIMAL;
		// We will sample directly from the color attachment
		image.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;

		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &offscreenPass_.color.image));
		vkGetImageMemoryRequirements(device, offscreenPass_.color.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &offscreenPass_.color.mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, offscreenPass_.color.image, offscreenPass_.color.mem, 0));

		VkImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
		colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		colorImageView.format = offscreenImageFormat;
		colorImageView.subresourceRange = {};
		colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		colorImageView.subresourceRange.baseMipLevel = 0;
		colorImageView.subresourceRange.levelCount = 1;
		colorImageView.subresourceRange.baseArrayLayer = 0;
		colorImageView.subresourceRange.layerCount = 1;
		colorImageView.image = offscreenPass_.color.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &colorImageView, nullptr, &offscreenPass_.color.view));

		// Create sampler to sample from the attachment in the fragment shader
		VkSamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeV = samplerInfo.addressModeU;
		samplerInfo.addressModeW = samplerInfo.addressModeU;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.maxAnisotropy = 1.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 1.0f;
		samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &offscreenPass_.sampler));

		// Depth stencil attachment
		image.format = fbDepthFormat;
		image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &offscreenPass_.depth.image));
		vkGetImageMemoryRequirements(device, offscreenPass_.depth.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &offscreenPass_.depth.mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, offscreenPass_.depth.image, offscreenPass_.depth.mem, 0));

		VkImageViewCreateInfo depthStencilView = vks::initializers::imageViewCreateInfo();
		depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		depthStencilView.format = fbDepthFormat;
		depthStencilView.flags = 0;
		depthStencilView.subresourceRange = {};
		depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		if (fbDepthFormat >= VK_FORMAT_D16_UNORM_S8_UINT)
			depthStencilView.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
		depthStencilView.subresourceRange.baseMipLevel = 0;
		depthStencilView.subresourceRange.levelCount = 1;
		depthStencilView.subresourceRange.baseArrayLayer = 0;
		depthStencilView.subresourceRange.layerCount = 1;
		depthStencilView.image = offscreenPass_.depth.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &depthStencilView, nullptr, &offscreenPass_.depth.view));

		// Create a separate render pass for the offscreen rendering as it may differ from the one used for scene rendering

		std::array<VkAttachmentDescription, 2> attchmentDescriptions = {};
		// Color attachment
		attchmentDescriptions[0].format = offscreenImageFormat;
		attchmentDescriptions[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attchmentDescriptions[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attchmentDescriptions[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attchmentDescriptions[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attchmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attchmentDescriptions[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attchmentDescriptions[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		// Depth attachment
		attchmentDescriptions[1].format = fbDepthFormat;
		attchmentDescriptions[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attchmentDescriptions[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attchmentDescriptions[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attchmentDescriptions[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attchmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attchmentDescriptions[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attchmentDescriptions[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
		VkAttachmentReference depthReference = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

		VkSubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDescription.colorAttachmentCount = 1;
		subpassDescription.pColorAttachments = &colorReference;
		subpassDescription.pDepthStencilAttachment = &depthReference;

		// Use subpass dependencies for layout transitions
		std::array<VkSubpassDependency, 2> dependencies{};

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		// Create the actual renderpass
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attchmentDescriptions.size());
		renderPassInfo.pAttachments = attchmentDescriptions.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDescription;
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &offscreenPass_.renderPass));

		VkImageView attachments[2]{};
		attachments[0] = offscreenPass_.color.view;
		attachments[1] = offscreenPass_.depth.view;

		VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
		fbufCreateInfo.renderPass = offscreenPass_.renderPass;
		fbufCreateInfo.attachmentCount = 2;
		fbufCreateInfo.pAttachments = attachments;
		fbufCreateInfo.width = offscreenPass_.width;
		fbufCreateInfo.height = offscreenPass_.height;
		fbufCreateInfo.layers = 1;

		VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &offscreenPass_.frameBuffer));

		// Fill a descriptor for later use in a descriptor set
		offscreenPass_.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		offscreenPass_.descriptor.imageView = offscreenPass_.color.view;
		offscreenPass_.descriptor.sampler = offscreenPass_.sampler;
	}

	void loadAssets()
	{
		scene.loadFromFile(getAssetPath() + "models/glowsphere.gltf", vulkanDevice, queue, vkglTF::FileLoadingFlags::PreTransformVertices | vkglTF::FileLoadingFlags::PreMultiplyVertexColors | vkglTF::FileLoadingFlags::FlipY);
		gradientTexture.loadFromFile(getAssetPath() + "textures/particle_gradient_rgba.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue);
	}

	void setupDescriptors()
	{
		// Pool
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, maxConcurrentFrames * 3),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maxConcurrentFrames * 3)
		};
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, maxConcurrentFrames * 2);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

		// Layouts
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;
		VkDescriptorSetLayoutCreateInfo descriptorLayout;

		// Scene rendering
		setLayoutBindings = {
			// Binding 0: Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
			// Binding 1: Fragment shader image sampler
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
			// Binding 2: Fragment shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 2)
		};
		descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayouts_.scene));

		// Fullscreen radial blur
		setLayoutBindings = {
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
			// Binding 1: Fragment shader image sampler
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1)
		};
		descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayouts_.radialBlur));

		// Sets per frame, just like the buffers themselves
		// Images do not need to be duplicated per frame, we reuse the same one for each frame
		VkDescriptorSetAllocateInfo allocInfo{};
		for (auto i = 0; i < uniformBuffers_.size(); i++) {
			// Scene rendering
			allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts_.scene, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets_[i].scene));

			std::vector<VkWriteDescriptorSet> offScreenWriteDescriptorSets = {
				// Binding 0: Vertex shader uniform buffer
				vks::initializers::writeDescriptorSet(descriptorSets_[i].scene, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers_[i].scene.descriptor),
				// Binding 1: Color gradient sampler
				vks::initializers::writeDescriptorSet(descriptorSets_[i].scene, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &gradientTexture.descriptor),
			};
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(offScreenWriteDescriptorSets.size()), offScreenWriteDescriptorSets.data(), 0, nullptr);

			// Fullscreen radial blur
			allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts_.radialBlur, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets_[i].radialBlur));

			std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
				// Binding 0: Vertex shader uniform buffer
				vks::initializers::writeDescriptorSet(descriptorSets_[i].radialBlur, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers_[i].blurParams.descriptor),
				// Binding 1: Fragment shader texture sampler
				vks::initializers::writeDescriptorSet(descriptorSets_[i].radialBlur, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,	1, &offscreenPass_.descriptor),
			};
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
		}
	}

	void preparePipelines()
	{
		// Layouts
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts_.scene, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts_.scene));

		pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts_.radialBlur, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts_.radialBlur));

		// Pipelines
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayouts_.radialBlur, renderPass, 0);
		pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
		pipelineCI.pRasterizationState = &rasterizationStateCI;
		pipelineCI.pColorBlendState = &colorBlendStateCI;
		pipelineCI.pMultisampleState = &multisampleStateCI;
		pipelineCI.pViewportState = &viewportStateCI;
		pipelineCI.pDepthStencilState = &depthStencilStateCI;
		pipelineCI.pDynamicState = &dynamicStateCI;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();

		// Radial blur pipeline
		shaderStages[0] = loadShader(getShadersPath() + "radialblur/radialblur.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "radialblur/radialblur.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		// Empty vertex input state (the vertex shader generates a screen covering triangle)
		VkPipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		pipelineCI.pVertexInputState = &emptyInputState;
		pipelineCI.layout = pipelineLayouts_.radialBlur;
		// Additive blending
		blendAttachmentState.colorWriteMask = 0xF;
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines_.radialBlur));

		// No blending (for debug display)
		blendAttachmentState.blendEnable = VK_FALSE;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines_.offscreenDisplay));

		// Phong pass
		pipelineCI.layout = pipelineLayouts_.scene;
		shaderStages[0] = loadShader(getShadersPath() + "radialblur/phongpass.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "radialblur/phongpass.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		blendAttachmentState.blendEnable = VK_FALSE;
		depthStencilStateCI.depthWriteEnable = VK_TRUE;
		pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position, vkglTF::VertexComponent::UV, vkglTF::VertexComponent::Color, vkglTF::VertexComponent::Normal });;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines_.phongPass));

		// Color only pass (offscreen blur base)
		shaderStages[0] = loadShader(getShadersPath() + "radialblur/colorpass.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "radialblur/colorpass.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		pipelineCI.renderPass = offscreenPass_.renderPass;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines_.colorPass));
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		for (auto& buffer : uniformBuffers_) {
			// Phong and color pass vertex shader uniform buffer
			VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer.scene, sizeof(UniformDataScene)));
			// Fullscreen radial blur parameters
			VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer.blurParams, sizeof(UniformDataBlurParams)));
			// Map persistent
			VK_CHECK_RESULT(buffer.scene.map());
			VK_CHECK_RESULT(buffer.blurParams.map());
		}
	}

	void updateUniformBuffers()
	{
		// Update uniform buffers for rendering the 3D scene
		uniformDataScene.projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 1.0f, 256.0f);
		camera.setRotation(camera.rotation + glm::vec3(0.0f, frameTimer * 10.0f, 0.0f));
		uniformDataScene.projection = camera.matrices.perspective;
		uniformDataScene.modelView = camera.matrices.view;
		// Add some animation to the post processing effect by moving through a color gradient for the radial blur
		if (!paused) {
			uniformDataScene.gradientPos += frameTimer * 0.1f;
		}
		memcpy(uniformBuffers_[currentBuffer].scene.mapped, &uniformDataScene, sizeof(UniformDataScene));
		// Update parameters for the radial blur pass
		memcpy(uniformBuffers_[currentBuffer].blurParams.mapped, &uniformDataBlurParams, sizeof(UniformDataBlurParams));
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		prepareOffscreen();
		prepareUniformBuffers();
		setupDescriptors();
		preparePipelines();
		prepared = true;
	}

	void buildCommandBuffer()
	{
		VkCommandBuffer cmdBuffer = drawCmdBuffers[currentBuffer];
		
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2]{};

		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

		/*
			First render pass: Offscreen rendering
		*/
		{
			clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
			clearValues[1].depthStencil = { 1.0f, 0 };

			VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
			renderPassBeginInfo.renderPass = offscreenPass_.renderPass;
			renderPassBeginInfo.framebuffer = offscreenPass_.frameBuffer;
			renderPassBeginInfo.renderArea.extent.width = offscreenPass_.width;
			renderPassBeginInfo.renderArea.extent.height = offscreenPass_.height;
			renderPassBeginInfo.clearValueCount = 2;
			renderPassBeginInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			VkViewport viewport = vks::initializers::viewport((float)offscreenPass_.width, (float)offscreenPass_.height, 0.0f, 1.0f);
			vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);
			VkRect2D scissor = vks::initializers::rect2D(offscreenPass_.width, offscreenPass_.height, 0, 0);
			vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);
			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.scene, 0, 1, &descriptorSets_[currentBuffer].scene, 0, nullptr);
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.colorPass);
			scene.draw(cmdBuffer);
			vkCmdEndRenderPass(cmdBuffer);
		}

		/*
			Note: Explicit synchronization is not required between the render pass, as this is done implicit via sub pass dependencies
		*/

		/*
			Second render pass: Scene rendering with applied radial blur
		*/
		{
			clearValues[0].color = defaultClearColor;
			clearValues[1].depthStencil = { 1.0f, 0 };

			VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
			renderPassBeginInfo.renderPass = renderPass;
			renderPassBeginInfo.framebuffer = frameBuffers[currentImageIndex];
			renderPassBeginInfo.renderArea.extent.width = width;
			renderPassBeginInfo.renderArea.extent.height = height;
			renderPassBeginInfo.clearValueCount = 2;
			renderPassBeginInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);
			VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);
			// 3D scene
			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.scene, 0, 1, &descriptorSets_[currentBuffer].scene, 0, nullptr);
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.phongPass);
			scene.draw(cmdBuffer);
			// Fullscreen triangle (clipped to a quad) with radial blur
			if (blur)
			{
				vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.radialBlur, 0, 1, &descriptorSets_[currentBuffer].radialBlur, 0, nullptr);
				vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, (displayTexture) ? pipelines_.offscreenDisplay : pipelines_.radialBlur);
				vkCmdDraw(cmdBuffer, 3, 1, 0, 0);
			}
			drawUI(cmdBuffer);
			vkCmdEndRenderPass(cmdBuffer);
		}

		VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
	}


	virtual void render()
	{
		if (!prepared)
			return;
		VulkanExampleBase::prepareFrame();
		updateUniformBuffers();
		buildCommandBuffer();
		VulkanExampleBase::submitFrame();
	}

	virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
	{
		if (overlay->header("Settings")) {
			overlay->checkBox("Radial blur", &blur);
			overlay->checkBox("Display render target only", &displayTexture);
			if (blur) {
				if (overlay->header("Blur parameters")) {
					overlay->sliderFloat("Scale", &uniformDataBlurParams.radialBlurScale, 0.1f, 1.0f);
					overlay->sliderFloat("Strength", &uniformDataBlurParams.radialBlurStrength, 0.1f, 2.0f);
					overlay->sliderFloat("Horiz. origin", &uniformDataBlurParams.radialOrigin.x, 0.0f, 1.0f);
					overlay->sliderFloat("Vert. origin", &uniformDataBlurParams.radialOrigin.y, 0.0f, 1.0f);
				}
			}
		}
	}
};

VULKAN_EXAMPLE_MAIN()
