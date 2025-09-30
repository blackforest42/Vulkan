/*
* Vulkan Example - Multisampling using resolve attachments (MSAA)
* 
* This sample shows how to do multisampled anti aliasing using built-in hardware via resolve attachments
* These are special attachments that a multi-sampled is resolved to using a fixed sample pattern
*
* Copyright (C) 2016-2025 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"

class VulkanExample : public VulkanExampleBase
{
public:
	bool useSampleShading = false;
	VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT;

	vkglTF::Model model;

	struct UniformData {
		glm::mat4 projection;
		glm::mat4 model;
		glm::vec4 lightPos = glm::vec4(5.0f, -5.0f, 5.0f, 1.0f);
	} uniformData;
	std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers_;

	struct {
		VkPipeline MSAA{ VK_NULL_HANDLE };
		VkPipeline MSAASampleShading{ VK_NULL_HANDLE };
	} pipelines_;

	VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
	VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
	std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSets_{};

	VkExtent2D attachmentSize{};

	// Holds the Vulkan resources required for the final multi sample output target
	struct MultiSampleTarget {
		struct {
			VkImage image{ VK_NULL_HANDLE };
			VkImageView view{ VK_NULL_HANDLE };
			VkDeviceMemory memory{ VK_NULL_HANDLE };
		} color;
		struct {
			VkImage image{ VK_NULL_HANDLE };
			VkImageView view{ VK_NULL_HANDLE };
			VkDeviceMemory memory{ VK_NULL_HANDLE };
		} depth;
	} multisampleTarget;

	VulkanExample() : VulkanExampleBase()
	{
		title = "Multisampling";
		camera_.type = Camera::CameraType::lookat;
		camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);
		camera_.setRotation(glm::vec3(0.0f, -90.0f, 0.0f));
		camera_.setTranslation(glm::vec3(2.5f, 2.5f, -7.5f));
	}

	~VulkanExample()
	{
		if (device_) {
			vkDestroyPipeline(device_, pipelines_.MSAA, nullptr);
			vkDestroyPipeline(device_, pipelines_.MSAASampleShading, nullptr);

			vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(device_, descriptorSetLayout, nullptr);

			// Destroy MSAA target
			vkDestroyImage(device_, multisampleTarget.color.image, nullptr);
			vkDestroyImageView(device_, multisampleTarget.color.view, nullptr);
			vkFreeMemory(device_, multisampleTarget.color.memory, nullptr);
			vkDestroyImage(device_, multisampleTarget.depth.image, nullptr);
			vkDestroyImageView(device_, multisampleTarget.depth.view, nullptr);
			vkFreeMemory(device_, multisampleTarget.depth.memory, nullptr);

			for (auto& buffer : uniformBuffers_) {
				buffer.destroy();
			}
		}
	}

	// Enable physical device features required for this example
	virtual void getEnabledFeatures()
	{
		// Enable sample rate shading filtering if supported
		if (deviceFeatures_.sampleRateShading) {
			enabledFeatures_.sampleRateShading = VK_TRUE;
		}
		// Enable anisotropic filtering if supported
		if (deviceFeatures_.samplerAnisotropy) {
			enabledFeatures_.samplerAnisotropy = VK_TRUE;
		}
	}

	// Creates a multi sample render target (image and view) that is used to resolve
	// into the visible frame buffer target in the render pass
	void setupMultisampleTarget()
	{
		// Check if device supports requested sample count for color and depth frame buffer
		assert((deviceProperties_.limits.framebufferColorSampleCounts & sampleCount) && (deviceProperties_.limits.framebufferDepthSampleCounts & sampleCount));

		// Color target
		VkImageCreateInfo info = vks::initializers::imageCreateInfo();
		info.imageType = VK_IMAGE_TYPE_2D;
		info.format = swapChain_.colorFormat;
		info.extent.width = width_;
		info.extent.height = height_;
		info.extent.depth = 1;
		info.mipLevels = 1;
		info.arrayLayers = 1;
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		info.tiling = VK_IMAGE_TILING_OPTIMAL;
		info.samples = sampleCount;
		// Image will only be used as a transient target
		info.usage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VK_CHECK_RESULT(vkCreateImage(device_, &info, nullptr, &multisampleTarget.color.image));

		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device_, multisampleTarget.color.image, &memReqs);
		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		memAlloc.allocationSize = memReqs.size;
		// We prefer a lazily allocated memory type
		// This means that the memory gets allocated when the implementation sees fit, e.g. when first using the images
		VkBool32 lazyMemTypePresent;
		memAlloc.memoryTypeIndex = vulkanDevice_->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, &lazyMemTypePresent);
		if (!lazyMemTypePresent)
		{
			// If this is not available, fall back to device local memory
			memAlloc.memoryTypeIndex = vulkanDevice_->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		}
		VK_CHECK_RESULT(vkAllocateMemory(device_, &memAlloc, nullptr, &multisampleTarget.color.memory));
		vkBindImageMemory(device_, multisampleTarget.color.image, multisampleTarget.color.memory, 0);

		// Create image view for the MSAA target
		VkImageViewCreateInfo viewInfo = vks::initializers::imageViewCreateInfo();
		viewInfo.image = multisampleTarget.color.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = swapChain_.colorFormat;
		viewInfo.components.r = VK_COMPONENT_SWIZZLE_R;
		viewInfo.components.g = VK_COMPONENT_SWIZZLE_G;
		viewInfo.components.b = VK_COMPONENT_SWIZZLE_B;
		viewInfo.components.a = VK_COMPONENT_SWIZZLE_A;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.layerCount = 1;

		VK_CHECK_RESULT(vkCreateImageView(device_, &viewInfo, nullptr, &multisampleTarget.color.view));

		// Depth target
		info.imageType = VK_IMAGE_TYPE_2D;
		info.format = depthFormat_;
		info.extent.width = width_;
		info.extent.height = height_;
		info.extent.depth = 1;
		info.mipLevels = 1;
		info.arrayLayers = 1;
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		info.tiling = VK_IMAGE_TILING_OPTIMAL;
		info.samples = sampleCount;
		// Image will only be used as a transient target
		info.usage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VK_CHECK_RESULT(vkCreateImage(device_, &info, nullptr, &multisampleTarget.depth.image));

		vkGetImageMemoryRequirements(device_, multisampleTarget.depth.image, &memReqs);
		memAlloc = vks::initializers::memoryAllocateInfo();
		memAlloc.allocationSize = memReqs.size;

		memAlloc.memoryTypeIndex = vulkanDevice_->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, &lazyMemTypePresent);
		if (!lazyMemTypePresent)
		{
			memAlloc.memoryTypeIndex = vulkanDevice_->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		}

		VK_CHECK_RESULT(vkAllocateMemory(device_, &memAlloc, nullptr, &multisampleTarget.depth.memory));
		vkBindImageMemory(device_, multisampleTarget.depth.image, multisampleTarget.depth.memory, 0);

		// Create image view for the MSAA target
		viewInfo.image = multisampleTarget.depth.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = depthFormat_;
		viewInfo.components.r = VK_COMPONENT_SWIZZLE_R;
		viewInfo.components.g = VK_COMPONENT_SWIZZLE_G;
		viewInfo.components.b = VK_COMPONENT_SWIZZLE_B;
		viewInfo.components.a = VK_COMPONENT_SWIZZLE_A;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		if (depthFormat_ >= VK_FORMAT_D16_UNORM_S8_UINT)
			viewInfo.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.layerCount = 1;

		VK_CHECK_RESULT(vkCreateImageView(device_, &viewInfo, nullptr, &multisampleTarget.depth.view));
	}

	// Setup a render pass for using a multi sampled attachment
	// and a resolve attachment that the msaa image is resolved
	// to at the end of the render pass
	void setupRenderPass()
	{
		// Overrides the virtual function of the base class
		
		attachmentSize = { width_, height_ };

		std::array<VkAttachmentDescription, 3> attachments = {};

		// Multisampled attachment that we render to
		attachments[0].format = swapChain_.colorFormat;
		attachments[0].samples = sampleCount;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// This is the frame buffer attachment to where the multisampled image
		// will be resolved to and which will be presented to the swapchain
		attachments[1].format = swapChain_.colorFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		// Multisampled depth attachment we render to
		attachments[2].format = depthFormat_;
		attachments[2].samples = sampleCount;
		attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[2].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorReference = {};
		colorReference.attachment = 0;
		colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthReference = {};
		depthReference.attachment = 2;
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		// Resolve attachment reference for the color attachment
		VkAttachmentReference resolveReference = {};
		resolveReference.attachment = 1;
		resolveReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorReference;
		// Pass our resolve attachments to the sub pass
		subpass.pResolveAttachments = &resolveReference;
		subpass.pDepthStencilAttachment = &depthReference;

		std::array<VkSubpassDependency, 2> dependencies{};

		// Depth attachment
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
		dependencies[0].dependencyFlags = 0;
		// Color attachment
		dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].dstSubpass = 0;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].srcAccessMask = 0;
		dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
		dependencies[1].dependencyFlags = 0;

		VkRenderPassCreateInfo renderPassInfo = vks::initializers::renderPassCreateInfo();
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 2;
		renderPassInfo.pDependencies = dependencies.data();

		VK_CHECK_RESULT(vkCreateRenderPass(device_, &renderPassInfo, nullptr, &renderPass_));
	}

	// Frame buffer attachments must match with render pass setup,
	// so we need to adjust frame buffer creation to cover our
	// multisample target
	void setupFrameBuffer()
	{
		// Overrides the virtual function of the base class

		// SRS - If the window is resized, the MSAA attachments need to be released and recreated
		if (attachmentSize.width != width_ || attachmentSize.height != height_)
		{
			attachmentSize = { width_, height_ };
			
			// Destroy MSAA target
			vkDestroyImage(device_, multisampleTarget.color.image, nullptr);
			vkDestroyImageView(device_, multisampleTarget.color.view, nullptr);
			vkFreeMemory(device_, multisampleTarget.color.memory, nullptr);
			vkDestroyImage(device_, multisampleTarget.depth.image, nullptr);
			vkDestroyImageView(device_, multisampleTarget.depth.view, nullptr);
			vkFreeMemory(device_, multisampleTarget.depth.memory, nullptr);
		}
		
		std::array<VkImageView, 3> attachments;

		setupMultisampleTarget();

		attachments[0] = multisampleTarget.color.view;
		// attachment[1] = swapchain image
		attachments[2] = multisampleTarget.depth.view;

		VkFramebufferCreateInfo frameBufferCreateInfo = {};
		frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frameBufferCreateInfo.pNext = NULL;
		frameBufferCreateInfo.renderPass = renderPass_;
		frameBufferCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		frameBufferCreateInfo.pAttachments = attachments.data();
		frameBufferCreateInfo.width = width_;
		frameBufferCreateInfo.height = height_;
		frameBufferCreateInfo.layers = 1;

		// Create frame buffers for every swap chain image
		frameBuffers_.resize(swapChain_.images_.size());
		for (uint32_t i = 0; i < frameBuffers_.size(); i++)
		{
			attachments[1] = swapChain_.imageViews_[i];
			VK_CHECK_RESULT(vkCreateFramebuffer(device_, &frameBufferCreateInfo, nullptr, &frameBuffers_[i]));
		}
	}

	void loadAssets()
	{
		model.loadFromFile(getAssetPath() + "models/voyager.gltf", vulkanDevice_, queue_, vkglTF::FileLoadingFlags::PreTransformVertices | vkglTF::FileLoadingFlags::FlipY);
	}

	void setupDescriptors()
	{
		// Pool
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_CONCURRENT_FRAMES),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_CONCURRENT_FRAMES),
		};
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, MAX_CONCURRENT_FRAMES);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo, nullptr, &descriptorPool_));

		// Layout
		const std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorLayout, nullptr, &descriptorSetLayout));

		// Sets per frame, just like the buffers themselves
		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool_, &descriptorSetLayout, 1);
		for (auto i = 0; i < uniformBuffers_.size(); i++) {
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSets_[i]));
			std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
				vks::initializers::writeDescriptorSet(descriptorSets_[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers_[i].descriptor),
			};
			vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
		}
	}

	void preparePipelines()
	{
		// Layout uses set 0 for passing vertex shader ubo and set 1 for fragment shader images (taken from glTF model)
		const std::vector<VkDescriptorSetLayout> setLayouts = {
			descriptorSetLayout,
			vkglTF::descriptorSetLayoutImage,
		};
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), 2);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

		// Pipeline
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

		// Setup multi sampling
		VkPipelineMultisampleStateCreateInfo multisampleState{};
		multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		// Number of samples to use for rasterization
		multisampleState.rasterizationSamples = sampleCount;

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass_, 0);
		pipelineCI.pInputAssemblyState = &inputAssemblyState;
		pipelineCI.pRasterizationState = &rasterizationState;
		pipelineCI.pColorBlendState = &colorBlendState;
		pipelineCI.pMultisampleState = &multisampleState;
		pipelineCI.pViewportState = &viewportState;
		pipelineCI.pDepthStencilState = &depthStencilState;
		pipelineCI.pDynamicState = &dynamicState;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();
		pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal, vkglTF::VertexComponent::UV, vkglTF::VertexComponent::Color });

		// MSAA rendering pipeline
		shaderStages[0] = loadShader(getShadersPath() + "multisampling/mesh.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "multisampling/mesh.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.MSAA));

		if (vulkanDevice_->features.sampleRateShading)
		{
			// MSAA with sample shading pipeline
			// Sample shading enables per-sample shading to avoid shader aliasing and smooth out e.g. high frequency texture maps
			// Note: This will trade performance for are more stable image
			
			// Enable per-sample shading (instead of per-fragment)
			multisampleState.sampleShadingEnable = VK_TRUE;
			// Minimum fraction for sample shading
			multisampleState.minSampleShading = 0.25f;
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.MSAASampleShading));
		}
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		for (auto& buffer : uniformBuffers_) {
			VK_CHECK_RESULT(vulkanDevice_->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer, sizeof(UniformData), &uniformData));
			VK_CHECK_RESULT(buffer.map());
		}
	}

	void updateUniformBuffers()
	{
		uniformData.projection = camera_.matrices.perspective;
		uniformData.model = camera_.matrices.view;
		memcpy(uniformBuffers_[currentBuffer_].mapped, &uniformData, sizeof(UniformData));
	}

	// Select the highest sample count usable by the platform
	// In a realworld application, this would be a user setting instead
	VkSampleCountFlagBits getMaxAvailableSampleCount()
	{
		VkSampleCountFlags supportedSampleCount = std::min(deviceProperties_.limits.framebufferColorSampleCounts, deviceProperties_.limits.framebufferDepthSampleCounts);
		std::vector< VkSampleCountFlagBits> possibleSampleCounts {
			VK_SAMPLE_COUNT_64_BIT, VK_SAMPLE_COUNT_32_BIT, VK_SAMPLE_COUNT_16_BIT, VK_SAMPLE_COUNT_8_BIT, VK_SAMPLE_COUNT_4_BIT, VK_SAMPLE_COUNT_2_BIT
		};
		for (auto& possibleSampleCount : possibleSampleCounts) {
			if (supportedSampleCount & possibleSampleCount) {
				return possibleSampleCount;
			}
		}
		return VK_SAMPLE_COUNT_1_BIT;
	}

	void prepare()
	{
		sampleCount = getMaxAvailableSampleCount();
		ui_.rasterizationSamples = sampleCount;
		VulkanExampleBase::prepare();
		loadAssets();
		prepareUniformBuffers();
		setupDescriptors();
		preparePipelines();
		prepared_ = true;
	}

	void buildCommandBuffer()
	{
		VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];
		
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[3]{};
		// Clear to a white background for higher contrast
		clearValues[0].color = { { 1.0f, 1.0f, 1.0f, 1.0f } };
		clearValues[1].color = { { 1.0f, 1.0f, 1.0f, 1.0f } };
		clearValues[2].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass_;
		renderPassBeginInfo.renderArea.extent.width = width_;
		renderPassBeginInfo.renderArea.extent.height = height_;
		renderPassBeginInfo.clearValueCount = 3;
		renderPassBeginInfo.pClearValues = clearValues;
		renderPassBeginInfo.framebuffer = frameBuffers_[currentImageIndex_];

		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

		vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		VkViewport viewport = vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
		vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

		VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
		vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

		vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets_[currentBuffer_], 0, nullptr);
		vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, useSampleShading ? pipelines_.MSAASampleShading : pipelines_.MSAA);
		model.draw(cmdBuffer, vkglTF::RenderFlags::BindImages, pipelineLayout);

		drawUI(cmdBuffer);

		vkCmdEndRenderPass(cmdBuffer);

		VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
	}

	virtual void render()
	{
		if (!prepared_)
			return;
		VulkanExampleBase::prepareFrame();
		updateUniformBuffers();
		buildCommandBuffer();
		VulkanExampleBase::submitFrame();

	}

	virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
	{
		if (vulkanDevice_->features.sampleRateShading) {
			if (overlay->header("Settings")) {
				overlay->checkBox("Sample rate shading", &useSampleShading);
			}
		}
	}
};

VULKAN_EXAMPLE_MAIN()
