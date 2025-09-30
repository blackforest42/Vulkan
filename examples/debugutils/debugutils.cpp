/*
* Vulkan Example - Example for the VK_EXT_debug_utils extension. Can be used in conjunction with a debugging app like RenderDoc (https://renderdoc.org)
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
	bool wireframe = true;
	bool glow = true;

	struct Models {
		vkglTF::Model scene, sceneGlow;
	} models_;

	struct UniformData {
		glm::mat4 projection;
		glm::mat4 model;
		glm::vec4 lightPos = glm::vec4(0.0f, 5.0f, 15.0f, 1.0f);
	} uniformData_;
	std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers_;

	struct Pipelines {
		VkPipeline toonshading;
		VkPipeline color;
		VkPipeline wireframe;
		VkPipeline postprocess;
	} pipelines_{};

	VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
	VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
	std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSets_{};

	// Framebuffer for offscreen rendering
	struct FrameBufferAttachment {
		VkImage image;
		VkDeviceMemory memory;
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

	// Function pointers for the VK_EXT_debug_utils_extension

	bool debugUtilsSupported = false;

	PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT{ nullptr };
	PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT{ nullptr };
	PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT{ nullptr };
	PFN_vkCmdInsertDebugUtilsLabelEXT vkCmdInsertDebugUtilsLabelEXT{ nullptr };
	PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT{ nullptr };
	PFN_vkQueueBeginDebugUtilsLabelEXT vkQueueBeginDebugUtilsLabelEXT{ nullptr };
	PFN_vkQueueInsertDebugUtilsLabelEXT vkQueueInsertDebugUtilsLabelEXT{ nullptr };
	PFN_vkQueueEndDebugUtilsLabelEXT vkQueueEndDebugUtilsLabelEXT{ nullptr };
	PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT{ nullptr };

	VulkanExample() : VulkanExampleBase()
	{
		title = "Debugging with VK_EXT_debug_utils";
		camera_.setRotation(glm::vec3(-4.35f, 16.25f, 0.0f));
		camera_.setRotationSpeed(0.5f);
		camera_.setPosition(glm::vec3(0.1f, 1.1f, -8.5f));
		camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 256.0f);
	}

	// Enable physical device features required for this example
	virtual void getEnabledFeatures()
	{
		// Fill mode non solid is required for wireframe display
		if (deviceFeatures_.fillModeNonSolid) {
			enabledFeatures_.fillModeNonSolid = VK_TRUE;
		};
		wireframe = deviceFeatures_.fillModeNonSolid;
	}

	~VulkanExample()
	{
		if (device_) {
			vkDestroyPipeline(device_, pipelines_.toonshading, nullptr);
			vkDestroyPipeline(device_, pipelines_.color, nullptr);
			vkDestroyPipeline(device_, pipelines_.postprocess, nullptr);
			if (pipelines_.wireframe != VK_NULL_HANDLE) {
				vkDestroyPipeline(device_, pipelines_.wireframe, nullptr);
			}
			vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(device_, descriptorSetLayout, nullptr);
			for (auto& buffer : uniformBuffers_) {
				buffer.destroy();
			}

			// Offscreen
			// Color attachment
			vkDestroyImageView(device_, offscreenPass_.color.view, nullptr);
			vkDestroyImage(device_, offscreenPass_.color.image, nullptr);
			vkFreeMemory(device_, offscreenPass_.color.memory, nullptr);

			// Depth attachment
			vkDestroyImageView(device_, offscreenPass_.depth.view, nullptr);
			vkDestroyImage(device_, offscreenPass_.depth.image, nullptr);
			vkFreeMemory(device_, offscreenPass_.depth.memory, nullptr);

			vkDestroyRenderPass(device_, offscreenPass_.renderPass, nullptr);
			vkDestroySampler(device_, offscreenPass_.sampler, nullptr);
			vkDestroyFramebuffer(device_, offscreenPass_.frameBuffer, nullptr);
		}
	}

	/*
		Debug utils functions
	*/

	// Checks if debug utils are supported (usually only when a graphics debugger is active) and does the setup necessary to use this debug utils
	void setupDebugUtils()
	{
		// Check if the debug utils extension is present (which is the case if run from a graphics debugger)
		bool extensionPresent = false;
		uint32_t extensionCount;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> extensions(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
		for (auto& extension : extensions) {
			if (strcmp(extension.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
				extensionPresent = true;
				break;
			}
		}

		if (extensionPresent) {
			// As with an other extension, function pointers need to be manually loaded
			vkCreateDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
			vkDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
			vkCmdBeginDebugUtilsLabelEXT = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(vkGetInstanceProcAddr(instance_, "vkCmdBeginDebugUtilsLabelEXT"));
			vkCmdInsertDebugUtilsLabelEXT = reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(vkGetInstanceProcAddr(instance_, "vkCmdInsertDebugUtilsLabelEXT"));
			vkCmdEndDebugUtilsLabelEXT = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(vkGetInstanceProcAddr(instance_, "vkCmdEndDebugUtilsLabelEXT"));
			vkQueueBeginDebugUtilsLabelEXT = reinterpret_cast<PFN_vkQueueBeginDebugUtilsLabelEXT>(vkGetInstanceProcAddr(instance_, "vkQueueBeginDebugUtilsLabelEXT"));
			vkQueueInsertDebugUtilsLabelEXT = reinterpret_cast<PFN_vkQueueInsertDebugUtilsLabelEXT>(vkGetInstanceProcAddr(instance_, "vkQueueInsertDebugUtilsLabelEXT"));
			vkQueueEndDebugUtilsLabelEXT = reinterpret_cast<PFN_vkQueueEndDebugUtilsLabelEXT>(vkGetInstanceProcAddr(instance_, "vkQueueEndDebugUtilsLabelEXT"));
			vkSetDebugUtilsObjectNameEXT = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(vkGetInstanceProcAddr(instance_, "vkSetDebugUtilsObjectNameEXT"));

			// Set flag if at least one function pointer is present
			debugUtilsSupported = (vkCreateDebugUtilsMessengerEXT != VK_NULL_HANDLE);
		}
		else {
			std::cout << "Warning: " << VK_EXT_DEBUG_UTILS_EXTENSION_NAME << " not present, debug utils are disabled.";
			std::cout << "Try running the sample from inside a Vulkan graphics debugger (e.g. RenderDoc)" << std::endl;
		}
	}

	// The debug utils extensions allows us to put labels into command buffers and queues (to e.g. mark regions of interest) and to name Vulkan objects
	// We wrap these into functions for convenience

	// Functions for putting labels into a command buffer
	// Labels consist of a name and an optional color
	// How or if these are diplayed depends on the debugger used (RenderDoc e.g. displays both)

	void cmdBeginLabel(VkCommandBuffer command_buffer, const char* label_name, std::vector<float> color)
	{
		if (!debugUtilsSupported) {
			return;
		}
		VkDebugUtilsLabelEXT label = { VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
		label.pLabelName = label_name;
		memcpy(label.color, color.data(), sizeof(float) * 4);
		vkCmdBeginDebugUtilsLabelEXT(command_buffer, &label);
	}

	void cmdInsertLabel(VkCommandBuffer command_buffer, const char* label_name, std::vector<float> color)
	{
		if (!debugUtilsSupported) {
			return;
		}
		VkDebugUtilsLabelEXT label = { VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
		label.pLabelName = label_name;
		memcpy(label.color, color.data(), sizeof(float) * 4);
		vkCmdInsertDebugUtilsLabelEXT(command_buffer, &label);
	}

	void cmdEndLabel(VkCommandBuffer command_buffer)
	{
		if (!debugUtilsSupported) {
			return;
		}
		vkCmdEndDebugUtilsLabelEXT(command_buffer);
	}

	// Functions for putting labels into a queue
	// Labels consist of a name and an optional color
	// How or if these are diplayed depends on the debugger used (RenderDoc e.g. displays both)

	void queueBeginLabel(VkQueue queue, const char* label_name, std::vector<float> color)
	{
		if (!debugUtilsSupported) {
			return;
		}
		VkDebugUtilsLabelEXT label = { VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
		label.pLabelName = label_name;
		memcpy(label.color, color.data(), sizeof(float) * 4);
		vkQueueBeginDebugUtilsLabelEXT(queue, &label);
	}

	void queueInsertLabel(VkQueue queue, const char* label_name, std::vector<float> color)
	{
		if (!debugUtilsSupported) {
			return;
		}
		VkDebugUtilsLabelEXT label = { VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
		label.pLabelName = label_name;
		memcpy(label.color, color.data(), sizeof(float) * 4);
		vkQueueInsertDebugUtilsLabelEXT(queue, &label);
	}

	void queueEndLabel(VkQueue queue)
	{
		if (!debugUtilsSupported) {
			return;
		}
		vkQueueEndDebugUtilsLabelEXT(queue);
	}

	// Function for naming Vulkan objects
	// In Vulkan, all objects (that can be named) are opaque unsigned 64 bit handles, and can be cased to uint64_t

	void setObjectName(VkDevice device, VkObjectType object_type, uint64_t object_handle, std::string object_name)
	{
		if (!debugUtilsSupported) {
			return;
		}
		VkDebugUtilsObjectNameInfoEXT name_info = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
		name_info.objectType = object_type;
		name_info.objectHandle = object_handle;
		name_info.pObjectName = object_name.c_str();
		vkSetDebugUtilsObjectNameEXT(device, &name_info);
	}

	// Prepare a texture target and framebuffer for offscreen rendering
	void prepareOffscreen()
	{
		const uint32_t dim = 256;
		const VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;

		offscreenPass_.width = 256;
		offscreenPass_.height = 256;

		// Find a suitable depth format
		VkFormat fbDepthFormat;
		VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice_, &fbDepthFormat);
		assert(validDepthFormat);

		// Color attachment
		VkImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = VK_IMAGE_TYPE_2D;
		image.format = colorFormat;
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
		VK_CHECK_RESULT(vkAllocateMemory(device_, &memAlloc, nullptr, &offscreenPass_.color.memory));
		VK_CHECK_RESULT(vkBindImageMemory(device_, offscreenPass_.color.image, offscreenPass_.color.memory, 0));

		VkImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
		colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		colorImageView.format = colorFormat;
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
		VK_CHECK_RESULT(vkCreateSampler(device_, &samplerInfo, nullptr, &offscreenPass_.sampler));

		// Depth stencil attachment
		image.format = fbDepthFormat;
		image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

		VK_CHECK_RESULT(vkCreateImage(device_, &image, nullptr, &offscreenPass_.depth.image));
		vkGetImageMemoryRequirements(device_, offscreenPass_.depth.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice_->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device_, &memAlloc, nullptr, &offscreenPass_.depth.memory));
		VK_CHECK_RESULT(vkBindImageMemory(device_, offscreenPass_.depth.image, offscreenPass_.depth.memory, 0));

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
		attchmentDescriptions[0].format = colorFormat;
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
		std::array<VkSubpassDependency, 2> dependencies;

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
		const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices | vkglTF::FileLoadingFlags::PreMultiplyVertexColors | vkglTF::FileLoadingFlags::FlipY;
		models_.scene.loadFromFile(getAssetPath() + "models/treasure_smooth.gltf", vulkanDevice_, queue_, glTFLoadingFlags);
		models_.sceneGlow.loadFromFile(getAssetPath() + "models/treasure_glow.gltf", vulkanDevice_, queue_, glTFLoadingFlags);
	}

	// We use a custom draw function so we can insert debug labels with the names of the glTF nodes
	void drawModel(vkglTF::Model &model, VkCommandBuffer cmdBuffer)
	{
		model.bindBuffers(cmdBuffer);
		for (auto i = 0; i < model.nodes.size(); i++)
		{
			// Insert a label for the current model's name
			cmdInsertLabel(cmdBuffer, model.nodes[i]->name.c_str(), { 0.0f, 0.0f, 0.0f, 0.0f });
			model.drawNode(model.nodes[i], cmdBuffer);
		}
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
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
			// Binding 1 : Fragment shader combined sampler
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorLayout, nullptr, &descriptorSetLayout));

		// Sets per frame, just like the buffers themselves
		// Images do not need to be duplicated per frame, we use the same for each descriptor to keep things simple
		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool_, &descriptorSetLayout, 1);
		for (auto i = 0; i < uniformBuffers_.size(); i++) {
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSets_[i]));
			std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
				// Binding 0 : Vertex shader uniform buffer
				vks::initializers::writeDescriptorSet(descriptorSets_[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers_[i].descriptor),
				// Binding 1 : Color map
				vks::initializers::writeDescriptorSet(descriptorSets_[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &offscreenPass_.descriptor)
			};
			vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
		}
	}

	void preparePipelines()
	{
		// Layout
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

		// Pipelines
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR	};
		VkPipelineDynamicStateCreateInfo dynamicStateCI =	vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass_);
		pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
		pipelineCI.pRasterizationState = &rasterizationStateCI;
		pipelineCI.pColorBlendState = &colorBlendStateCI;
		pipelineCI.pMultisampleState = &multisampleStateCI;
		pipelineCI.pViewportState = &viewportStateCI;
		pipelineCI.pDepthStencilState = &depthStencilStateCI;
		pipelineCI.pDynamicState = &dynamicStateCI;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();
		pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal, vkglTF::VertexComponent::UV, vkglTF::VertexComponent::Color});

		// Toon shading pipeline
		shaderStages[0] = loadShader(getShadersPath() + "debugutils/toon.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "debugutils/toon.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.toonshading));

		// Color only pipeline
		shaderStages[0] = loadShader(getShadersPath() + "debugutils/colorpass.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "debugutils/colorpass.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		pipelineCI.renderPass = offscreenPass_.renderPass;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.color));

		// Wire frame rendering pipeline
		if (deviceFeatures_.fillModeNonSolid)
		{
			rasterizationStateCI.polygonMode = VK_POLYGON_MODE_LINE;
			pipelineCI.renderPass = renderPass_;
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.wireframe));
		}

		// Post processing effect
		shaderStages[0] = loadShader(getShadersPath() + "debugutils/postprocess.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "debugutils/postprocess.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		depthStencilStateCI.depthTestEnable = VK_FALSE;
		depthStencilStateCI.depthWriteEnable = VK_FALSE;
		rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
		blendAttachmentState.colorWriteMask = 0xF;
		blendAttachmentState.blendEnable =  VK_TRUE;
		blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.postprocess));
	}

	// For convencience we name our Vulkan objects in a single place
	void nameDebugObjects()
	{
		// Name some objects for debugging
		setObjectName(device_, VK_OBJECT_TYPE_IMAGE, (uint64_t)offscreenPass_.color.image, "Off-screen color framebuffer");
		setObjectName(device_, VK_OBJECT_TYPE_IMAGE, (uint64_t)offscreenPass_.depth.image, "Off-screen depth framebuffer");
		setObjectName(device_, VK_OBJECT_TYPE_SAMPLER, (uint64_t)offscreenPass_.sampler, "Off-screen framebuffer default sampler");

		setObjectName(device_, VK_OBJECT_TYPE_BUFFER, (uint64_t)models_.scene.vertices.buffer, "Scene vertex buffer");
		setObjectName(device_, VK_OBJECT_TYPE_BUFFER, (uint64_t)models_.scene.indices.buffer, "Scene index buffer");
		setObjectName(device_, VK_OBJECT_TYPE_BUFFER, (uint64_t)models_.sceneGlow.vertices.buffer, "Glow vertex buffer");
		setObjectName(device_, VK_OBJECT_TYPE_BUFFER, (uint64_t)models_.sceneGlow.indices.buffer, "Glow index buffer");
		
		// Shader module count starts at 2 when UI overlay in base class is enabled
		uint32_t moduleIndex = settings_.overlay ? 2 : 0;
		setObjectName(device_, VK_OBJECT_TYPE_SHADER_MODULE, (uint64_t)shaderModules_[moduleIndex + 0], "Toon shading vertex shader");
		setObjectName(device_, VK_OBJECT_TYPE_SHADER_MODULE, (uint64_t)shaderModules_[moduleIndex + 1], "Toon shading fragment shader");
		setObjectName(device_, VK_OBJECT_TYPE_SHADER_MODULE, (uint64_t)shaderModules_[moduleIndex + 2], "Color-only vertex shader");
		setObjectName(device_, VK_OBJECT_TYPE_SHADER_MODULE, (uint64_t)shaderModules_[moduleIndex + 3], "Color-only fragment shader");
		setObjectName(device_, VK_OBJECT_TYPE_SHADER_MODULE, (uint64_t)shaderModules_[moduleIndex + 4], "Postprocess vertex shader");
		setObjectName(device_, VK_OBJECT_TYPE_SHADER_MODULE, (uint64_t)shaderModules_[moduleIndex + 5], "Postprocess fragment shader");

		setObjectName(device_, VK_OBJECT_TYPE_PIPELINE_LAYOUT, (uint64_t)pipelineLayout, "Shared pipeline layout");
		setObjectName(device_, VK_OBJECT_TYPE_PIPELINE, (uint64_t)pipelines_.toonshading, "Toon shading pipeline");
		setObjectName(device_, VK_OBJECT_TYPE_PIPELINE, (uint64_t)pipelines_.color, "Color only pipeline");
		if (deviceFeatures_.fillModeNonSolid) {
			setObjectName(device_, VK_OBJECT_TYPE_PIPELINE, (uint64_t)pipelines_.wireframe, "Wireframe rendering pipeline");
		}
		setObjectName(device_, VK_OBJECT_TYPE_PIPELINE, (uint64_t)pipelines_.postprocess, "Post processing pipeline");

		// Shared objects
		setObjectName(device_, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, (uint64_t)descriptorSetLayout, "Shared descriptor set layout");
		for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
			setObjectName(device_, VK_OBJECT_TYPE_BUFFER, (uint64_t)uniformBuffers_[i].buffer, "Scene uniform buffer block for frame " + std::to_string(i));
			setObjectName(device_, VK_OBJECT_TYPE_DESCRIPTOR_SET, (uint64_t)descriptorSets_[i], "Shared descriptor set for frame " + std::to_string(i));
		}
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		for (auto& buffer : uniformBuffers_) {
			VK_CHECK_RESULT(vulkanDevice_->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer, sizeof(UniformData), &uniformData_));
			VK_CHECK_RESULT(buffer.map());
		}
	}

	void updateUniformBuffers()
	{
		uniformData_.projection = camera_.matrices.perspective;
		uniformData_.model = camera_.matrices.view;
		memcpy(uniformBuffers_[currentBuffer_].mapped, &uniformData_, sizeof(uniformData_));
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		setupDebugUtils();
		loadAssets();
		prepareOffscreen();
		prepareUniformBuffers();
		setupDescriptors();
		preparePipelines();
		nameDebugObjects();
		prepared_ = true;
	}

	void buildCommandBuffer()
	{
		VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];
		

		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
		VkClearValue clearValues[2]{};

		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

		/*
			First render pass: Offscreen rendering
		*/
		if (glow)
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

			cmdBeginLabel(cmdBuffer, "Off-screen scene rendering", { 1.0f, 0.78f, 0.05f, 1.0f });

			vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = vks::initializers::viewport((float)offscreenPass_.width, (float)offscreenPass_.height, 0.0f, 1.0f);
			vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

			VkRect2D scissor = vks::initializers::rect2D(offscreenPass_.width, offscreenPass_.height, 0, 0);
			vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets_[currentBuffer_], 0, nullptr);
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.color);

			drawModel(models_.sceneGlow, cmdBuffer);

			vkCmdEndRenderPass(cmdBuffer);

			cmdEndLabel(cmdBuffer);
		}

		/*
			Note: Explicit synchronization is not required between the render pass, as this is done implicit via sub pass dependencies
		*/

		/*
			Second render pass: Scene rendering with applied bloom
		*/
		{
			clearValues[0].color = defaultClearColor;
			clearValues[1].depthStencil = { 1.0f, 0 };

			VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
			renderPassBeginInfo.renderPass = renderPass_;
			renderPassBeginInfo.framebuffer = frameBuffers_[currentImageIndex_];
			renderPassBeginInfo.renderArea.extent.width = width_;
			renderPassBeginInfo.renderArea.extent.height = height_;
			renderPassBeginInfo.clearValueCount = 2;
			renderPassBeginInfo.pClearValues = clearValues;

			cmdBeginLabel(cmdBuffer, "Render scene", { 0.5f, 0.76f, 0.34f, 1.0f });

			vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
			vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

			VkRect2D scissor = vks::initializers::rect2D(wireframe ? width_ / 2 : width_, height_, 0, 0);
			vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets_[currentBuffer_], 0, nullptr);

			// Solid rendering

			cmdBeginLabel(cmdBuffer, "Toon shading draw", { 0.78f, 0.74f, 0.9f, 1.0f });

			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.toonshading);
			drawModel(models_.scene, cmdBuffer);

			cmdEndLabel(cmdBuffer);

			// Wireframe rendering
			if (wireframe)
			{
				cmdBeginLabel(cmdBuffer, "Wireframe draw", { 0.53f, 0.78f, 0.91f, 1.0f });

				scissor.offset.x = width_ / 2;
				vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

				vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.wireframe);
				drawModel(models_.scene, cmdBuffer);

				cmdEndLabel(cmdBuffer);

				scissor.offset.x = 0;
				scissor.extent.width = width_;
				vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);
			}

			// Post processing
			if (glow)
			{
				cmdBeginLabel(cmdBuffer, "Apply post processing", { 0.93f, 0.89f, 0.69f, 1.0f });

				vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.postprocess);
				// Full screen quad is generated by the vertex shaders, so we reuse four vertices (for four invocations) from current vertex buffer
				vkCmdDraw(cmdBuffer, 4, 1, 0, 0);

				cmdEndLabel(cmdBuffer);
			}

			cmdBeginLabel(cmdBuffer, "UI overlay", { 0.23f, 0.65f, 0.28f, 1.0f });
			drawUI(cmdBuffer);
			cmdEndLabel(cmdBuffer);

			vkCmdEndRenderPass(cmdBuffer);

			cmdEndLabel(cmdBuffer);

		}

		VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
	}

	virtual void render()
	{
		if (!prepared_)
			return;
		queueBeginLabel(queue_, "Graphics queue command buffer submission", { 1.0f, 1.0f, 1.0f, 1.0f });
		VulkanExampleBase::prepareFrame();
		updateUniformBuffers();
		buildCommandBuffer();
		VulkanExampleBase::submitFrame();
		queueEndLabel(queue_);
	}

	virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
	{
		if (overlay->header("Info")) {
			overlay->text("VK_EXT_debug_utils %s", (debugUtilsSupported? "supported" : "not supported"));
		}
		if (overlay->header("Settings")) {
			overlay->checkBox("Glow", &glow);
			if (deviceFeatures_.fillModeNonSolid) {
				overlay->checkBox("Wireframe", &wireframe);
			}
		}
	}
};

VULKAN_EXAMPLE_MAIN()
