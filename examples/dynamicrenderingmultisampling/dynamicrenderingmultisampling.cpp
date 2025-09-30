/*
 * Vulkan Example - Using Multi sampling with VK_KHR_dynamic_rendering
 *
 * Copyright (C) 2025 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"

class VulkanExample : public VulkanExampleBase
{
public:
	PFN_vkCmdBeginRenderingKHR vkCmdBeginRenderingKHR{ VK_NULL_HANDLE };
	PFN_vkCmdEndRenderingKHR vkCmdEndRenderingKHR{ VK_NULL_HANDLE };

	VkPhysicalDeviceDynamicRenderingFeaturesKHR enabledDynamicRenderingFeaturesKHR{};

	vkglTF::Model model;

	const VkSampleCountFlagBits multiSampleCount = VK_SAMPLE_COUNT_4_BIT;

	struct UniformData {
		glm::mat4 projection;
		glm::mat4 modelView;
		glm::vec4 viewPos;
	} uniformData;
	std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers_;

	VkPipeline pipeline{ VK_NULL_HANDLE };
	VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
	VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
	std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSets_{};

	// Intermediate images used for multi sampling
	struct Image {
		VkImage image{ VK_NULL_HANDLE };
		VkImageView view{ VK_NULL_HANDLE };
		VkDeviceMemory memory{ VK_NULL_HANDLE };
	};
	Image renderImage;

	VulkanExample() : VulkanExampleBase()
	{
		title = "Multi sampling with dynamic rendering";
		camera_.type = Camera::CameraType::lookat;
		camera_.setPosition(glm::vec3(0.0f, 0.0f, -10.0f));
		camera_.setRotation(glm::vec3(-7.5f, 72.0f, 0.0f));
		camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);
		settings_.overlay = false;

		enabledInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

		// The sample uses the extension (instead of Vulkan 1.2, where dynamic rendering is core)
		enabledDeviceExtensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_KHR_MAINTENANCE2_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);

		// in addition to the extension, the feature needs to be explicitly enabled too by chaining the extension structure into device creation
		enabledDynamicRenderingFeaturesKHR.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;
		enabledDynamicRenderingFeaturesKHR.dynamicRendering = VK_TRUE;

		deviceCreatepNextChain_ = &enabledDynamicRenderingFeaturesKHR;
	}

	~VulkanExample() override
	{
		if (device_) {
			vkDestroyPipeline(device_, pipeline, nullptr);
			vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(device_, descriptorSetLayout, nullptr);
			vkDestroyImage(device_, renderImage.image, nullptr);
			vkDestroyImageView(device_, renderImage.view, nullptr);
			vkFreeMemory(device_, renderImage.memory, nullptr);
			for (auto& buffer : uniformBuffers_) {
				buffer.destroy();
			}
		}
	}

	void setupRenderPass() override
	{
		// With VK_KHR_dynamic_rendering we no longer need a render pass, so we can skip the sample base render pass setup
		renderPass_ = VK_NULL_HANDLE;
	}

	void setupFrameBuffer() override
	{
		// With VK_KHR_dynamic_rendering we no longer need a frame buffer, so we can so skip the sample base framebuffer setup
		// For multi sampling we need intermediate images that are then resolved to the final presentation image 
		vkDestroyImage(device_, renderImage.image, nullptr);
		vkDestroyImageView(device_, renderImage.view, nullptr);
		vkFreeMemory(device_, renderImage.memory, nullptr);
		VkImageCreateInfo renderImageCI = vks::initializers::imageCreateInfo();
		renderImageCI.imageType = VK_IMAGE_TYPE_2D;
		renderImageCI.format = swapChain_.colorFormat;
		renderImageCI.extent = { width_, height_, 1 };
		renderImageCI.mipLevels = 1;
		renderImageCI.arrayLayers = 1;
		renderImageCI.samples = multiSampleCount;
		renderImageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		renderImageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		renderImageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VK_CHECK_RESULT(vkCreateImage(device_, &renderImageCI, nullptr, &renderImage.image));
		VkMemoryRequirements memReqs{};
		vkGetImageMemoryRequirements(device_, renderImage.image, &memReqs);
		VkMemoryAllocateInfo memAllloc{};
		memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAllloc.allocationSize = memReqs.size;
		memAllloc.memoryTypeIndex = vulkanDevice_->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device_, &memAllloc, nullptr, &renderImage.memory));
		VK_CHECK_RESULT(vkBindImageMemory(device_, renderImage.image, renderImage.memory, 0));
		VkImageViewCreateInfo imageViewCI = vks::initializers::imageViewCreateInfo();
		imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCI.image = renderImage.image;
		imageViewCI.format = swapChain_.colorFormat;
		imageViewCI.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		VK_CHECK_RESULT(vkCreateImageView(device_, &imageViewCI, nullptr, &renderImage.view));
	}

	// We need to override the default depth/stencil setup to create a depth image that supports multi sampling
	void setupDepthStencil() override
	{
		VkImageCreateInfo imageCI{};
		imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCI.imageType = VK_IMAGE_TYPE_2D;
		imageCI.format = depthFormat_;
		imageCI.extent = { width_, height_, 1 };
		imageCI.mipLevels = 1;
		imageCI.arrayLayers = 1;
		imageCI.samples = multiSampleCount;
		imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		VK_CHECK_RESULT(vkCreateImage(device_, &imageCI, nullptr, &depthStencil_.image));
		VkMemoryRequirements memReqs{};
		vkGetImageMemoryRequirements(device_, depthStencil_.image, &memReqs);
		VkMemoryAllocateInfo memAllloc{};
		memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAllloc.allocationSize = memReqs.size;
		memAllloc.memoryTypeIndex = vulkanDevice_->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device_, &memAllloc, nullptr, &depthStencil_.memory));
		VK_CHECK_RESULT(vkBindImageMemory(device_, depthStencil_.image, depthStencil_.memory, 0));
		VkImageViewCreateInfo depthImageViewCI{};
		depthImageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		depthImageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		depthImageViewCI.image = depthStencil_.image;
		depthImageViewCI.format = depthFormat_;
		depthImageViewCI.subresourceRange.baseMipLevel = 0;
		depthImageViewCI.subresourceRange.levelCount = 1;
		depthImageViewCI.subresourceRange.baseArrayLayer = 0;
		depthImageViewCI.subresourceRange.layerCount = 1;
		depthImageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		// Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
		if (depthFormat_ >= VK_FORMAT_D16_UNORM_S8_UINT) {
			depthImageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
		}
		VK_CHECK_RESULT(vkCreateImageView(device_, &depthImageViewCI, nullptr, &depthStencil_.view));
	}

	// Enable physical device features required for this example
	void getEnabledFeatures() override
	{
		// Enable anisotropic filtering if supported
		if (deviceFeatures_.samplerAnisotropy) {
			enabledFeatures_.samplerAnisotropy = VK_TRUE;
		};
	}

	void loadAssets()
	{
		const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices | vkglTF::FileLoadingFlags::PreMultiplyVertexColors | vkglTF::FileLoadingFlags::FlipY;
		model.loadFromFile(getAssetPath() + "models/voyager.gltf", vulkanDevice_, queue_, glTFLoadingFlags);
	}

	void setupDescriptors()
	{	
		// Pool
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_CONCURRENT_FRAMES),
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
		// Layout
		// Uses set 0 for passing vertex shader ubo and set 1 for fragment shader images (taken from glTF model)
		const std::vector<VkDescriptorSetLayout> setLayouts = {
			descriptorSetLayout,
			vkglTF::descriptorSetLayoutImage,
		};
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), 2);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

		// Pipeline
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(multiSampleCount, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

		// We no longer need to set a renderpass for the pipeline create info
		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo();
		pipelineCI.layout = pipelineLayout;
		pipelineCI.pInputAssemblyState = &inputAssemblyState;
		pipelineCI.pRasterizationState = &rasterizationState;
		pipelineCI.pColorBlendState = &colorBlendState;
		pipelineCI.pMultisampleState = &multisampleState;
		pipelineCI.pViewportState = &viewportState;
		pipelineCI.pDepthStencilState = &depthStencilState;
		pipelineCI.pDynamicState = &dynamicState;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();
		pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal, vkglTF::VertexComponent::UV });

		// New create info to define color, depth and stencil attachments at pipeline create time
		VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo{};
		pipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
		pipelineRenderingCreateInfo.colorAttachmentCount = 1;
		pipelineRenderingCreateInfo.pColorAttachmentFormats = &swapChain_.colorFormat;
		pipelineRenderingCreateInfo.depthAttachmentFormat = depthFormat_;
		pipelineRenderingCreateInfo.stencilAttachmentFormat = depthFormat_;
		// Chain into the pipeline creat einfo
		pipelineCI.pNext = &pipelineRenderingCreateInfo;

		shaderStages[0] = loadShader(getShadersPath() + "dynamicrendering/texture.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "dynamicrendering/texture.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipeline));
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
		uniformData.modelView = camera_.matrices.view;
		uniformData.viewPos = camera_.viewPos;
		memcpy(uniformBuffers_[currentBuffer_].mapped, &uniformData, sizeof(uniformData));
	}

	void prepare() override
	{
		VulkanExampleBase::prepare();

		// Since we use an extension, we need to expliclity load the function pointers for extension related Vulkan commands
		vkCmdBeginRenderingKHR = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(vkGetDeviceProcAddr(device_, "vkCmdBeginRenderingKHR"));
		vkCmdEndRenderingKHR = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(vkGetDeviceProcAddr(device_, "vkCmdEndRenderingKHR"));

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
		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

		// With dynamic rendering there are no subpass dependencies, so we need to take care of proper layout transitions by using barriers
		// This set of barriers prepares the color and depth images for output
		vks::tools::insertImageMemoryBarrier(
			cmdBuffer,
			renderImage.image,
			0,
			VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });
		vks::tools::insertImageMemoryBarrier(
			cmdBuffer,
			depthStencil_.image,
			0,
			VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1 });
		// We also need an initial barrier for the swap chain image to make it usable for general layout
		vks::tools::insertImageMemoryBarrier(
			cmdBuffer,
			swapChain_.images_[currentImageIndex_],
			0,
			VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		// New structures are used to define the attachments used in dynamic rendering
		VkRenderingAttachmentInfoKHR colorAttachment{};
		colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
		colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.clearValue.color = { 0.0f,0.0f,0.0f,0.0f };
		// When multi sampling is used, we use intermediate images to render and resolve to the swap chain images
		colorAttachment.imageView = renderImage.view;
		colorAttachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
		colorAttachment.resolveImageView = swapChain_.imageViews_[currentImageIndex_];
		colorAttachment.resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL;

		// A single depth stencil attachment info can be used, but they can also be specified separately.
		// When both are specified separately, the only requirement is that the image view is identical.			
		VkRenderingAttachmentInfoKHR depthStencilAttachment{};
		depthStencilAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
		depthStencilAttachment.imageView = depthStencil_.view;
		depthStencilAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		depthStencilAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthStencilAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		depthStencilAttachment.clearValue.depthStencil = { 1.0f,  0 };

		VkRenderingInfoKHR renderingInfo{};
		renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
		renderingInfo.renderArea = { 0, 0, width_, height_ };
		renderingInfo.layerCount = 1;
		renderingInfo.colorAttachmentCount = 1;
		renderingInfo.pColorAttachments = &colorAttachment;
		renderingInfo.pDepthAttachment = &depthStencilAttachment;
		renderingInfo.pStencilAttachment = &depthStencilAttachment;

		// Begin dynamic rendering
		vkCmdBeginRenderingKHR(cmdBuffer, &renderingInfo);

		VkViewport viewport = vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
		vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

		VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
		vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

		vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets_[currentBuffer_], 0, nullptr);
		vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		model.draw(cmdBuffer, vkglTF::RenderFlags::BindImages, pipelineLayout);

		drawUI(cmdBuffer);

		// End dynamic rendering
		vkCmdEndRenderingKHR(cmdBuffer);

		// This set of barriers prepares the color image for presentation, we don't need to care for the depth image
		vks::tools::insertImageMemoryBarrier(
			cmdBuffer,
			swapChain_.images_[currentImageIndex_],
			VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			0,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
	}

	void render() override
	{
		if (!prepared_)
			return;
		VulkanExampleBase::prepareFrame();
		updateUniformBuffers();
		buildCommandBuffer();
		VulkanExampleBase::submitFrame();
	}
};

VULKAN_EXAMPLE_MAIN()
