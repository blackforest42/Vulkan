/*
* Vulkan Example - Conservative rasterization
*
* Note: Requires a device that supports the VK_EXT_conservative_rasterization extension
*
* Uses an offscreen buffer with lower resolution to demonstrate the effect of conservative rasterization
*
* Copyright (C) 2016-2025 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"

class VulkanExample : public VulkanExampleBase
{
public:
	// Fetch and store conservative rasterization state props for display purposes
	VkPhysicalDeviceConservativeRasterizationPropertiesEXT conservativeRasterProps{};

	bool conservativeRasterEnabled = true;

	struct Vertex {
		float position[3];
		float color[3];
	};

	struct Triangle {
		vks::Buffer vertices;
		vks::Buffer indices;
		uint32_t indexCount{ 0 };
	} triangle;

	struct UniformData {
		glm::mat4 projection;
		glm::mat4 model;
	} uniformData_;
	std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers_;

	struct PipelineLayouts {
		VkPipelineLayout scene{ VK_NULL_HANDLE };
		VkPipelineLayout fullscreen{ VK_NULL_HANDLE };
	} pipelineLayouts_;

	struct Pipelines {
		VkPipeline triangle{ VK_NULL_HANDLE };
		VkPipeline triangleConservativeRaster{ VK_NULL_HANDLE };
		VkPipeline triangleOverlay{ VK_NULL_HANDLE };
		VkPipeline fullscreen{ VK_NULL_HANDLE };
	} pipelines_;

	struct DescriptorSetLayouts {
		VkDescriptorSetLayout scene{ VK_NULL_HANDLE };
		VkDescriptorSetLayout fullscreen{ VK_NULL_HANDLE };
	} descriptorSetLayouts_;

	struct DescriptorSets {
		VkDescriptorSet scene{ VK_NULL_HANDLE };
		VkDescriptorSet fullscreen{ VK_NULL_HANDLE };
	};
	std::array<DescriptorSets, MAX_CONCURRENT_FRAMES> descriptorSets_;

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

	VulkanExample() : VulkanExampleBase()
	{
		title = "Conservative rasterization";
		camera_.type = Camera::CameraType::lookat;
		camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 512.0f);
		camera_.setRotation(glm::vec3(0.0f));
		camera_.setTranslation(glm::vec3(0.0f, 0.0f, -2.0f));

		// Enable extension required for conservative rasterization
		enabledDeviceExtensions_.push_back(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
		// Reading device properties of conservative rasterization requires VK_KHR_get_physical_device_properties2 to be enabled
		enabledInstanceExtensions_.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
	}

	~VulkanExample()
	{
		if (device_) {
			vkDestroyImageView(device_, offscreenPass_.color.view, nullptr);
			vkDestroyImage(device_, offscreenPass_.color.image, nullptr);
			vkFreeMemory(device_, offscreenPass_.color.mem, nullptr);
			vkDestroyImageView(device_, offscreenPass_.depth.view, nullptr);
			vkDestroyImage(device_, offscreenPass_.depth.image, nullptr);
			vkFreeMemory(device_, offscreenPass_.depth.mem, nullptr);
			vkDestroyRenderPass(device_, offscreenPass_.renderPass, nullptr);
			vkDestroySampler(device_, offscreenPass_.sampler, nullptr);
			vkDestroyFramebuffer(device_, offscreenPass_.frameBuffer, nullptr);
			vkDestroyPipeline(device_, pipelines_.triangle, nullptr);
			vkDestroyPipeline(device_, pipelines_.triangleOverlay, nullptr);
			vkDestroyPipeline(device_, pipelines_.triangleConservativeRaster, nullptr);
			vkDestroyPipeline(device_, pipelines_.fullscreen, nullptr);
			vkDestroyPipelineLayout(device_, pipelineLayouts_.fullscreen, nullptr);
			vkDestroyPipelineLayout(device_, pipelineLayouts_.scene, nullptr);
			vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.scene, nullptr);
			vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.fullscreen, nullptr);
			triangle.vertices.destroy();
			triangle.indices.destroy();
			for (auto& buffer : uniformBuffers_) {
				buffer.destroy();
			}
		}
	}

	void getEnabledFeatures()
	{
		enabledFeatures_.fillModeNonSolid = deviceFeatures_.fillModeNonSolid;
		enabledFeatures_.wideLines = deviceFeatures_.wideLines;
	}

	/*
		Setup offscreen framebuffer, attachments and render passes for lower resolution rendering of the scene
	*/
	void prepareOffscreen()
	{
		// We "magnify" the offscreen rendered triangle so that the conservative rasterization feature is easier to see
		const int32_t magnification = 16;

		offscreenPass_.width = width_ / magnification;
		offscreenPass_.height = height_ / magnification;

		const VkFormat fbColorFormat = VK_FORMAT_R8G8B8A8_UNORM;

		// Find a suitable depth format
		VkFormat fbDepthFormat;
		VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice_, &fbDepthFormat);
		assert(validDepthFormat);

		// Color attachment
		VkImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = VK_IMAGE_TYPE_2D;
		image.format = fbColorFormat;
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

		VK_CHECK_RESULT(vkCreateImage(device_, &image, nullptr, &offscreenPass_.color.image));
		vkGetImageMemoryRequirements(device_, offscreenPass_.color.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice_->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device_, &memAlloc, nullptr, &offscreenPass_.color.mem));
		VK_CHECK_RESULT(vkBindImageMemory(device_, offscreenPass_.color.image, offscreenPass_.color.mem, 0));

		VkImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
		colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		colorImageView.format = fbColorFormat;
		colorImageView.subresourceRange = {};
		colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		colorImageView.subresourceRange.baseMipLevel = 0;
		colorImageView.subresourceRange.levelCount = 1;
		colorImageView.subresourceRange.baseArrayLayer = 0;
		colorImageView.subresourceRange.layerCount = 1;
		colorImageView.image = offscreenPass_.color.image;
		VK_CHECK_RESULT(vkCreateImageView(device_, &colorImageView, nullptr, &offscreenPass_.color.view));

		// Create sampler to sample from the attachment in the fragment shader
		VkSamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();
		samplerInfo.magFilter = VK_FILTER_NEAREST;
		samplerInfo.minFilter = VK_FILTER_NEAREST;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeV = samplerInfo.addressModeU;
		samplerInfo.addressModeW = samplerInfo.addressModeU;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.maxAnisotropy = 1.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 1.0f;
		samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VK_CHECK_RESULT(vkCreateSampler(device_, &samplerInfo, nullptr, &offscreenPass_.sampler));

		// Depth stencil attachment
		image.format = fbDepthFormat;
		image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

		VK_CHECK_RESULT(vkCreateImage(device_, &image, nullptr, &offscreenPass_.depth.image));
		vkGetImageMemoryRequirements(device_, offscreenPass_.depth.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice_->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device_, &memAlloc, nullptr, &offscreenPass_.depth.mem));
		VK_CHECK_RESULT(vkBindImageMemory(device_, offscreenPass_.depth.image, offscreenPass_.depth.mem, 0));

		VkImageViewCreateInfo depthStencilView = vks::initializers::imageViewCreateInfo();
		depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		depthStencilView.format = fbDepthFormat;
		depthStencilView.flags = 0;
		depthStencilView.subresourceRange = {};
		depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
		depthStencilView.subresourceRange.baseMipLevel = 0;
		depthStencilView.subresourceRange.levelCount = 1;
		depthStencilView.subresourceRange.baseArrayLayer = 0;
		depthStencilView.subresourceRange.layerCount = 1;
		depthStencilView.image = offscreenPass_.depth.image;
		VK_CHECK_RESULT(vkCreateImageView(device_, &depthStencilView, nullptr, &offscreenPass_.depth.view));

		// Create a separate render pass for the offscreen rendering as it may differ from the one used for scene rendering

		std::array<VkAttachmentDescription, 2> attchmentDescriptions = {};
		// Color attachment
		attchmentDescriptions[0].format = fbColorFormat;
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
		std::array<VkSubpassDependency, 3> dependencies{};

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
		dependencies[0].dependencyFlags = 0;

		dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].dstSubpass = 0;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[2].srcSubpass = 0;
		dependencies[2].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[2].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[2].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[2].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[2].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		// Create the actual renderpass
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attchmentDescriptions.size());
		renderPassInfo.pAttachments = attchmentDescriptions.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDescription;
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();

		VK_CHECK_RESULT(vkCreateRenderPass(device_, &renderPassInfo, nullptr, &offscreenPass_.renderPass));

		VkImageView attachments[2];
		attachments[0] = offscreenPass_.color.view;
		attachments[1] = offscreenPass_.depth.view;

		VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
		fbufCreateInfo.renderPass = offscreenPass_.renderPass;
		fbufCreateInfo.attachmentCount = 2;
		fbufCreateInfo.pAttachments = attachments;
		fbufCreateInfo.width = offscreenPass_.width;
		fbufCreateInfo.height = offscreenPass_.height;
		fbufCreateInfo.layers = 1;

		VK_CHECK_RESULT(vkCreateFramebuffer(device_, &fbufCreateInfo, nullptr, &offscreenPass_.frameBuffer));

		// Fill a descriptor for later use in a descriptor set
		offscreenPass_.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		offscreenPass_.descriptor.imageView = offscreenPass_.color.view;
		offscreenPass_.descriptor.sampler = offscreenPass_.sampler;
	}
		
	void loadAssets()
	{
		// Create a single triangle
		struct Vertex {
			float position[3];
			float color[3];
		};

		std::vector<Vertex> vertexBuffer = {
			{ {  1.0f,  1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
			{ { -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
			{ {  0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } }
		};
		uint32_t vertexBufferSize = static_cast<uint32_t>(vertexBuffer.size()) * sizeof(Vertex);
		std::vector<uint32_t> indexBuffer = { 0, 1, 2 };
		triangle.indexCount = static_cast<uint32_t>(indexBuffer.size());
		uint32_t indexBufferSize = triangle.indexCount * sizeof(uint32_t);

		struct StagingBuffers {
			vks::Buffer vertices;
			vks::Buffer indices;
		} stagingBuffers;

		// Host visible source buffers (staging)
		VK_CHECK_RESULT(vulkanDevice_->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&stagingBuffers.vertices,
			vertexBufferSize,
			vertexBuffer.data()));
		VK_CHECK_RESULT(vulkanDevice_->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&stagingBuffers.indices,
			indexBufferSize,
			indexBuffer.data()));

		// Device local destination buffers
		VK_CHECK_RESULT(vulkanDevice_->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&triangle.vertices,
			vertexBufferSize));
		VK_CHECK_RESULT(vulkanDevice_->createBuffer(
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&triangle.indices,
			indexBufferSize));

		// Copy from host do device
		vulkanDevice_->copyBuffer(&stagingBuffers.vertices, &triangle.vertices, queue_);
		vulkanDevice_->copyBuffer(&stagingBuffers.indices, &triangle.indices, queue_);

		// Clean up
		stagingBuffers.vertices.destroy();
		stagingBuffers.indices.destroy();
	}

	void setupDescriptors()
	{
		// Pool
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_CONCURRENT_FRAMES * 2),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_CONCURRENT_FRAMES * 2)
		};
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, MAX_CONCURRENT_FRAMES * 2);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo, nullptr, &descriptorPool_));

		// Layouts
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;
		VkDescriptorSetLayoutCreateInfo descriptorLayout;

		// Scene rendering
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),			// Binding 0: Vertex shader uniform buffer
		};
		descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorLayout, nullptr, &descriptorSetLayouts_.scene));

		// Fullscreen pass
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0)	// Binding 0: Fragment shader image sampler
		};
		descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorLayout, nullptr, &descriptorSetLayouts_.fullscreen));

		// Sets per frame, just like the buffers themselves
		// Images do not need to be duplicated per frame, we reuse the same one for each frame
		for (auto i = 0; i < uniformBuffers_.size(); i++) {
			// Scene rendering
			VkDescriptorSetAllocateInfo descriptorSetAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool_, &descriptorSetLayouts_.scene, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &descriptorSetAllocInfo, &descriptorSets_[i].scene));
			std::vector<VkWriteDescriptorSet> offScreenWriteDescriptorSets = {
				vks::initializers::writeDescriptorSet(descriptorSets_[i].scene, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers_[i].descriptor),
			};
			vkUpdateDescriptorSets(device_, static_cast<uint32_t>(offScreenWriteDescriptorSets.size()), offScreenWriteDescriptorSets.data(), 0, nullptr);

			// Fullscreen pass
			descriptorSetAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool_, &descriptorSetLayouts_.fullscreen, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &descriptorSetAllocInfo, &descriptorSets_[i].fullscreen));
			std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
				vks::initializers::writeDescriptorSet(descriptorSets_[i].fullscreen, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &offscreenPass_.descriptor),
			};
			vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
		}
	}

	void preparePipelines()
	{
		// Layouts
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
		pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts_.scene, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts_.scene));

		pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts_.fullscreen, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts_.fullscreen));

		// Pipelines
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1,	&blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, };
		VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

		VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_CLOCKWISE, 0);

		/*
			Conservative rasterization setup
		*/

		/*
			Get device properties for conservative rasterization
			Requires VK_KHR_get_physical_device_properties2 and manual function pointer creation
		*/
		PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2KHR>(vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceProperties2KHR"));
		assert(vkGetPhysicalDeviceProperties2KHR);
		VkPhysicalDeviceProperties2KHR deviceProps2{};
		conservativeRasterProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT;
		deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
		deviceProps2.pNext = &conservativeRasterProps;
		vkGetPhysicalDeviceProperties2KHR(physicalDevice_, &deviceProps2);

		// Vertex bindings and attributes
		std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
		};
		std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0),					// Location 0: Position
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3),	// Location 1: Color
		};
		VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

		VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(pipelineLayouts_.fullscreen, renderPass_, 0);
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateCI;
		pipelineCreateInfo.pRasterizationState = &rasterizationStateCI;
		pipelineCreateInfo.pColorBlendState = &colorBlendStateCI;
		pipelineCreateInfo.pMultisampleState = &multisampleStateCI;
		pipelineCreateInfo.pViewportState = &viewportStateCI;
		pipelineCreateInfo.pDepthStencilState = &depthStencilStateCI;
		pipelineCreateInfo.pDynamicState = &dynamicStateCI;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();

		// Full screen pass
		shaderStages[0] = loadShader(getShadersPath() + "conservativeraster/fullscreen.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "conservativeraster/fullscreen.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		// Empty vertex input state (full screen triangle generated in vertex shader)
		VkPipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		pipelineCreateInfo.pVertexInputState = &emptyInputState;
		pipelineCreateInfo.layout = pipelineLayouts_.fullscreen;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineCreateInfo, nullptr, &pipelines_.fullscreen));

		pipelineCreateInfo.pVertexInputState = &vertexInputState;
		pipelineCreateInfo.layout = pipelineLayouts_.scene;

		// Original triangle outline
		// TODO: Check support for lines
		rasterizationStateCI.lineWidth = 2.0f;
		rasterizationStateCI.polygonMode = VK_POLYGON_MODE_LINE;
		shaderStages[0] = loadShader(getShadersPath() + "conservativeraster/triangle.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "conservativeraster/triangleoverlay.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineCreateInfo, nullptr, &pipelines_.triangleOverlay));

		pipelineCreateInfo.renderPass = offscreenPass_.renderPass;

		/*
			Triangle rendering
		*/
		rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
		shaderStages[0] = loadShader(getShadersPath() + "conservativeraster/triangle.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "conservativeraster/triangle.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

		/*
			Basic pipeline
		*/
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineCreateInfo, nullptr, &pipelines_.triangle));

		/*
			Pipeline with conservative rasterization enabled
		*/
		VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeRasterStateCI{};
		conservativeRasterStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
		conservativeRasterStateCI.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
		conservativeRasterStateCI.extraPrimitiveOverestimationSize = conservativeRasterProps.maxExtraPrimitiveOverestimationSize;

		// Conservative rasterization state has to be chained into the pipeline rasterization state create info structure
		rasterizationStateCI.pNext = &conservativeRasterStateCI;

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineCreateInfo, nullptr, &pipelines_.triangleConservativeRaster));
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		for (auto& buffer : uniformBuffers_) {
			// Scene matrices uniform buffer
			VK_CHECK_RESULT(vulkanDevice_->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer, sizeof(UniformData)));
			VK_CHECK_RESULT(buffer.map());
		}
	}

	void updateUniformBuffers()
	{
		uniformData_.projection = camera_.matrices.perspective;
		uniformData_.model = camera_.matrices.view;
		memcpy(uniformBuffers_[currentBuffer_].mapped, &uniformData_, sizeof(UniformData));
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		prepareOffscreen();
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

		/*
			First render pass: Render a low res triangle to an offscreen framebuffer to use for visualization in second pass
		*/
		{
			VkClearValue clearValues[2]{};
			clearValues[0].color = { { 0.25f, 0.25f, 0.25f, 0.0f } };
			clearValues[1].depthStencil = { 1.0f, 0 };

			VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
			renderPassBeginInfo.renderPass = offscreenPass_.renderPass;
			renderPassBeginInfo.framebuffer = offscreenPass_.frameBuffer;
			renderPassBeginInfo.renderArea.extent.width = offscreenPass_.width;
			renderPassBeginInfo.renderArea.extent.height = offscreenPass_.height;
			renderPassBeginInfo.clearValueCount = 2;
			renderPassBeginInfo.pClearValues = clearValues;

			VkViewport viewport = vks::initializers::viewport((float)offscreenPass_.width, (float)offscreenPass_.height, 0.0f, 1.0f);
			vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

			VkRect2D scissor = vks::initializers::rect2D(offscreenPass_.width, offscreenPass_.height, 0, 0);
			vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

			vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.scene, 0, 1, &descriptorSets_[currentBuffer_].scene, 0, nullptr);
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, conservativeRasterEnabled ? pipelines_.triangleConservativeRaster : pipelines_.triangle);

			VkDeviceSize offsets[1] = { 0 };
			vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &triangle.vertices.buffer, offsets);
			vkCmdBindIndexBuffer(cmdBuffer, triangle.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);
			vkCmdDrawIndexed(cmdBuffer, triangle.indexCount, 1, 0, 0, 0);

			vkCmdEndRenderPass(cmdBuffer);
		}

		/*
			Note: Explicit synchronization is not required between the render pass, as this is done implicit via sub pass dependencies
		*/

		/*
			Second render pass: Render scene with conservative rasterization
		*/
		{
			VkClearValue clearValues[2];
			clearValues[0].color = { { 0.25f, 0.25f, 0.25f, 0.25f } };
			clearValues[1].depthStencil = { 1.0f, 0 };

			VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
			renderPassBeginInfo.framebuffer = frameBuffers_[currentImageIndex_];
			renderPassBeginInfo.renderPass = renderPass_;
			renderPassBeginInfo.renderArea.offset.x = 0;
			renderPassBeginInfo.renderArea.offset.y = 0;
			renderPassBeginInfo.renderArea.extent.width = width_;
			renderPassBeginInfo.renderArea.extent.height = height_;
			renderPassBeginInfo.clearValueCount = 2;
			renderPassBeginInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			VkViewport viewport = vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
			vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);
			VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
			vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

			// Low-res triangle from offscreen framebuffer
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.fullscreen);
			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.fullscreen, 0, 1, &descriptorSets_[currentBuffer_].fullscreen, 0, nullptr);
			vkCmdDraw(cmdBuffer, 3, 1, 0, 0);

			// Overlay actual triangle
			VkDeviceSize offsets[1] = { 0 };
			vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &triangle.vertices.buffer, offsets);
			vkCmdBindIndexBuffer(cmdBuffer, triangle.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.triangleOverlay);
			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.scene, 0, 1, &descriptorSets_[currentBuffer_].scene, 0, nullptr);
			vkCmdDraw(cmdBuffer, 3, 1, 0, 0);

			drawUI(cmdBuffer);

			vkCmdEndRenderPass(cmdBuffer);
		}

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
		if (overlay->header("Settings")) {
			overlay->checkBox("Conservative rasterization", &conservativeRasterEnabled);
		}
		if (overlay->header("Device properties")) {
			overlay->text("maxExtraPrimitiveOverestimationSize: %f", conservativeRasterProps.maxExtraPrimitiveOverestimationSize);
			overlay->text("extraPrimitiveOverestimationSizeGranularity: %f", conservativeRasterProps.extraPrimitiveOverestimationSizeGranularity);
			overlay->text("primitiveUnderestimation:  %s", conservativeRasterProps.primitiveUnderestimation ? "yes" : "no");
			overlay->text("conservativePointAndLineRasterization:  %s", conservativeRasterProps.conservativePointAndLineRasterization ? "yes" : "no");
			overlay->text("degenerateTrianglesRasterized: %s", conservativeRasterProps.degenerateTrianglesRasterized ? "yes" : "no");
			overlay->text("degenerateLinesRasterized: %s", conservativeRasterProps.degenerateLinesRasterized ? "yes" : "no");
			overlay->text("fullyCoveredFragmentShaderInputVariable: %s", conservativeRasterProps.fullyCoveredFragmentShaderInputVariable ? "yes" : "no");
			overlay->text("conservativeRasterizationPostDepthCoverage: %s", conservativeRasterProps.conservativeRasterizationPostDepthCoverage ? "yes" : "no");
		}

	}
};

VULKAN_EXAMPLE_MAIN()
