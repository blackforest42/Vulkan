/*
* Vulkan Example - Order Independent Transparency rendering using linked lists
*
* Copyright by Sascha Willems - www.saschawillems.de
* Copyright by Daemyung Jang  - dm86.jang@gmail.com
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"

#define NODE_COUNT 20

class VulkanExample : public VulkanExampleBase
{
public:
	struct {
		vkglTF::Model sphere;
		vkglTF::Model cube;
	} models_;

	struct Node {
		glm::vec4 color;
		float depth{ 0.0f };
		uint32_t next{ 0 };
	};

	struct {
		uint32_t count{ 0 };
		uint32_t maxNodeCount{ 0 };
	} geometrySBO;

	struct GeometryPass {
		VkRenderPass renderPass{ VK_NULL_HANDLE };
		VkFramebuffer framebuffer{ VK_NULL_HANDLE };
		vks::Buffer geometry;
		vks::Texture headIndex;
		vks::Buffer linkedList;
	} geometryPass;

	struct RenderPassUniformData {
		glm::mat4 projection;
		glm::mat4 view;
	} renderPassUniformData;
	std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> renderPassUniformBuffer;

	struct ObjectData {
		glm::mat4 model;
		glm::vec4 color;
	};

	struct {
		VkDescriptorSetLayout geometry{ VK_NULL_HANDLE };
		VkDescriptorSetLayout color{ VK_NULL_HANDLE };
	} descriptorSetLayouts_;

	struct {
		VkPipelineLayout geometry{ VK_NULL_HANDLE };
		VkPipelineLayout color{ VK_NULL_HANDLE };
	} pipelineLayouts_;

	struct {
		VkPipeline geometry{ VK_NULL_HANDLE };
		VkPipeline color{ VK_NULL_HANDLE };
	} pipelines_;

	struct DescriptorSets {
		VkDescriptorSet geometry{ VK_NULL_HANDLE };
		VkDescriptorSet color{ VK_NULL_HANDLE };
	};
	std::array<DescriptorSets, MAX_CONCURRENT_FRAMES> descriptorSets_{};

	VkDeviceSize objectUniformBufferSize{ 0 };

	VulkanExample() : VulkanExampleBase()
	{
		title_ = "Order independent transparency rendering";
		camera_.type_ = Camera::CameraType::lookat;
		camera_.setPosition(glm::vec3(0.0f, 0.0f, -6.0f));
		camera_.setRotation(glm::vec3(0.0f, 0.0f, 0.0f));
		camera_.setPerspective(60.0f, (float) width_ / (float) height_, 0.1f, 256.0f);
	}

	~VulkanExample()
	{
		if (device_) {
			vkDestroyPipeline(device_, pipelines_.geometry, nullptr);
			vkDestroyPipeline(device_, pipelines_.color, nullptr);
			vkDestroyPipelineLayout(device_, pipelineLayouts_.geometry, nullptr);
			vkDestroyPipelineLayout(device_, pipelineLayouts_.color, nullptr);
			vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.geometry, nullptr);
			vkDestroyDescriptorSetLayout(device_, descriptorSetLayouts_.color, nullptr);
			destroyGeometryPass();
			for (auto& buffer : renderPassUniformBuffer) {
				buffer.destroy();
			}
		}
	}

	void getEnabledFeatures() override
	{
		// The linked lists are built in a fragment shader using atomic stores, so the sample won't work without that feature available
		if (deviceFeatures_.fragmentStoresAndAtomics) {
			enabledFeatures_.fragmentStoresAndAtomics = VK_TRUE;
		} else {
			vks::tools::exitFatal("Selected GPU does not support stores and atomic operations in the fragment stage", VK_ERROR_FEATURE_NOT_PRESENT);
		}
	};

	void loadAssets()
	{
		const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices | vkglTF::FileLoadingFlags::FlipY;
		models_.sphere.loadFromFile(getAssetPath() + "models/sphere.gltf", vulkanDevice_, queue_, glTFLoadingFlags);
		models_.cube.loadFromFile(getAssetPath() + "models/cube.gltf", vulkanDevice_, queue_, glTFLoadingFlags);
	}

	void prepareUniformBuffers()
	{
		for (auto& buffer : renderPassUniformBuffer) {
			// Create an uniform buffer for a render pass.
			VK_CHECK_RESULT(vulkanDevice_->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer, sizeof(RenderPassUniformData)));
			VK_CHECK_RESULT(buffer.map());
		}
	}

	void prepareGeometryPass()
	{
		VkSubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

		// Geometry render pass doesn't need any output attachment.
		VkRenderPassCreateInfo renderPassInfo = vks::initializers::renderPassCreateInfo();
		renderPassInfo.attachmentCount = 0;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDescription;

		VK_CHECK_RESULT(vkCreateRenderPass(device_, &renderPassInfo, nullptr, &geometryPass.renderPass));

		// Geometry frame buffer doesn't need any output attachment.
		VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
		fbufCreateInfo.renderPass = geometryPass.renderPass;
		fbufCreateInfo.attachmentCount = 0;
		fbufCreateInfo.width = width_;
		fbufCreateInfo.height = height_;
		fbufCreateInfo.layers = 1;

		VK_CHECK_RESULT(vkCreateFramebuffer(device_, &fbufCreateInfo, nullptr, &geometryPass.framebuffer));

		// Create a buffer for GeometrySBO
		vks::Buffer stagingBuffer;
	
		VK_CHECK_RESULT(vulkanDevice_->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&stagingBuffer,
			sizeof(geometrySBO)));
		VK_CHECK_RESULT(stagingBuffer.map());

		VK_CHECK_RESULT(vulkanDevice_->createBuffer(
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&geometryPass.geometry,
			sizeof(geometrySBO)));

		// Set up GeometrySBO data.
		geometrySBO.count = 0;
		geometrySBO.maxNodeCount = NODE_COUNT * width_ * height_;
		memcpy(stagingBuffer.mapped, &geometrySBO, sizeof(geometrySBO));

		// Copy data to device
		VkCommandBuffer copyCmd = vulkanDevice_->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = sizeof(geometrySBO);
		vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, geometryPass.geometry.buffer, 1, &copyRegion);
		vulkanDevice_->flushCommandBuffer(copyCmd, queue_, true);

		stagingBuffer.destroy();
		
		// Create a texture for HeadIndex.
		// This image will track the head index of each fragment.
		geometryPass.headIndex.device = vulkanDevice_;

		VkImageCreateInfo imageInfo = vks::initializers::imageCreateInfo();
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = VK_FORMAT_R32_UINT;
		imageInfo.extent.width = width_;
		imageInfo.extent.height = height_;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
#if (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK) || defined(VK_USE_PLATFORM_METAL_EXT))
		// SRS - On macOS/iOS use linear tiling for atomic image access, see https://github.com/KhronosGroup/MoltenVK/issues/1027
		imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
#else
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
#endif
		imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

		VK_CHECK_RESULT(vkCreateImage(device_, &imageInfo, nullptr, &geometryPass.headIndex.image));

		geometryPass.headIndex.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device_, geometryPass.headIndex.image, &memReqs);

		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice_->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		VK_CHECK_RESULT(vkAllocateMemory(device_, &memAlloc, nullptr, &geometryPass.headIndex.deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device_, geometryPass.headIndex.image, geometryPass.headIndex.deviceMemory, 0));

		VkImageViewCreateInfo imageViewInfo = vks::initializers::imageViewCreateInfo();
		imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewInfo.format = VK_FORMAT_R32_UINT;
		imageViewInfo.flags = 0;
		imageViewInfo.image = geometryPass.headIndex.image;
		imageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageViewInfo.subresourceRange.baseMipLevel = 0;
		imageViewInfo.subresourceRange.levelCount = 1;
		imageViewInfo.subresourceRange.baseArrayLayer = 0;
		imageViewInfo.subresourceRange.layerCount = 1;

		VK_CHECK_RESULT(vkCreateImageView(device_, &imageViewInfo, nullptr, &geometryPass.headIndex.view));

		geometryPass.headIndex.width = width_;
		geometryPass.headIndex.height = height_;
		geometryPass.headIndex.mipLevels = 1;
		geometryPass.headIndex.layerCount = 1;
		geometryPass.headIndex.descriptor.imageView = geometryPass.headIndex.view;
		geometryPass.headIndex.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		geometryPass.headIndex.sampler = VK_NULL_HANDLE;

		// Create a buffer for LinkedListSBO
		VK_CHECK_RESULT(vulkanDevice_->createBuffer(
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&geometryPass.linkedList,
			sizeof(Node) * geometrySBO.maxNodeCount));

		// Change HeadIndex image's layout from UNDEFINED to GENERAL
		VkCommandBufferAllocateInfo cmdBufAllocInfo = vks::initializers::commandBufferAllocateInfo(cmdPool_, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);

		VkCommandBuffer cmdBuf;
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device_, &cmdBufAllocInfo, &cmdBuf));

		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuf, &cmdBufInfo));

		VkImageMemoryBarrier barrier = vks::initializers::imageMemoryBarrier();
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		barrier.image = geometryPass.headIndex.image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.layerCount = 1;

		vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

		VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuf));

		VkSubmitInfo submitInfo = vks::initializers::submitInfo();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cmdBuf;

		VK_CHECK_RESULT(vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE));
		VK_CHECK_RESULT(vkQueueWaitIdle(queue_));
	}

	void setupDescriptors()
	{
		// Pool
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_CONCURRENT_FRAMES),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, MAX_CONCURRENT_FRAMES),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_CONCURRENT_FRAMES * 3),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, MAX_CONCURRENT_FRAMES * 2),
		};
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, MAX_CONCURRENT_FRAMES * 2);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo, nullptr, &descriptorPool_));

		// Layouts

		// Create a geometry descriptor set layout
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// renderPassUniformData
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0),
			// AtomicSBO
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
			// headIndexImage
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT, 2),
			// LinkedListSBO
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 3),
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorLayoutCI, nullptr, &descriptorSetLayouts_.geometry));

		// Create a color descriptor set layout
		setLayoutBindings = {
			// headIndexImage
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
			// LinkedListSBO
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
		};
		descriptorLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorLayoutCI, nullptr, &descriptorSetLayouts_.color));

		updateDescriptors();
	}

	void updateDescriptors()
	{
		// Sets per frame, just like the buffers themselves
		// Images and GPU-only SSBO do not need to be duplicated per frame, we reuse the same one for each frame
		for (auto i = 0; i < renderPassUniformBuffer.size(); i++) {
			// Images and linked buffers are recreated on resize and part of the descriptors, so we need to update those at runtime
			VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool_, &descriptorSetLayouts_.geometry, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSets_[i].geometry));
			std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
				// Binding 0: renderPassUniformData
				vks::initializers::writeDescriptorSet(descriptorSets_[i].geometry, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &renderPassUniformBuffer[i].descriptor),
				// Binding 2: GeometrySBO
				vks::initializers::writeDescriptorSet(descriptorSets_[i].geometry, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &geometryPass.geometry.descriptor),
				// Binding 3: headIndexImage
				vks::initializers::writeDescriptorSet(descriptorSets_[i].geometry, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2, &geometryPass.headIndex.descriptor),
				// Binding 4: LinkedListSBO
				vks::initializers::writeDescriptorSet(descriptorSets_[i].geometry, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &geometryPass.linkedList.descriptor)
			};
			vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

			// Update a color descriptor set
			allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool_, &descriptorSetLayouts_.color, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSets_[i].color));
			writeDescriptorSets = {
				// Binding 0: headIndexImage
				vks::initializers::writeDescriptorSet(descriptorSets_[i].color, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0, &geometryPass.headIndex.descriptor),
				// Binding 1: LinkedListSBO
				vks::initializers::writeDescriptorSet(descriptorSets_[i].color, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &geometryPass.linkedList.descriptor)
			};
			vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
		}
	}

	void preparePipelines()
	{
		// Layouts

		// Create a geometry pipeline layout
		VkPipelineLayoutCreateInfo pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts_.geometry, 1);
		// Static object data passed using push constants
		VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(ObjectData), 0);
		pipelineLayoutCI.pushConstantRangeCount = 1;
		pipelineLayoutCI.pPushConstantRanges = &pushConstantRange;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr, &pipelineLayouts_.geometry));

		// Create a color pipeline layout
		pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts_.color, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCI, nullptr, &pipelineLayouts_.color));

		// Pipelines
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(0, nullptr);
		VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayouts_.geometry, geometryPass.renderPass);
		pipelineCI.pInputAssemblyState = &inputAssemblyState;
		pipelineCI.pRasterizationState = &rasterizationState;
		pipelineCI.pColorBlendState = &colorBlendState;
		pipelineCI.pMultisampleState = &multisampleState;
		pipelineCI.pViewportState = &viewportState;
		pipelineCI.pDepthStencilState = &depthStencilState;
		pipelineCI.pDynamicState = &dynamicState;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();
		pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position });

		// Create a geometry pipeline
		shaderStages[0] = loadShader(getShadersPath() + "oit/geometry.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "oit/geometry.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.geometry));

		// Create a color pipeline
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayouts_.color, renderPass_);
		pipelineCI.pInputAssemblyState = &inputAssemblyState;
		pipelineCI.pRasterizationState = &rasterizationState;
		pipelineCI.pColorBlendState = &colorBlendState;
		pipelineCI.pMultisampleState = &multisampleState;
		pipelineCI.pViewportState = &viewportState;
		pipelineCI.pDepthStencilState = &depthStencilState;
		pipelineCI.pDynamicState = &dynamicState;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();
		pipelineCI.pVertexInputState = &vertexInputInfo;

		shaderStages[0] = loadShader(getShadersPath() + "oit/color.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "oit/color.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;
		rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.color));
	}

	void updateUniformBuffers()
	{
		renderPassUniformData.projection = camera_.matrices_.perspective;
		renderPassUniformData.view = camera_.matrices_.view;
		memcpy(renderPassUniformBuffer[currentBuffer_].mapped, &renderPassUniformData, sizeof(RenderPassUniformData));
	}

	void prepare() override
	{
		VulkanExampleBase::prepare();
		loadAssets();
		prepareUniformBuffers();
		prepareGeometryPass();
		setupDescriptors();
		preparePipelines();
		prepared_ = true;
	}

	void buildCommandBuffer()
	{
		VkCommandBuffer cmdBuffer = drawCmdBuffers_[currentBuffer_];
		
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width_;
		renderPassBeginInfo.renderArea.extent.height = height_;

		VkViewport viewport = vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
		VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);

		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

		// Update dynamic viewport state
		vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

		// Update dynamic scissor state
		vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

		VkClearColorValue clearColor;
		clearColor.uint32[0] = 0xffffffff;

		VkImageSubresourceRange subresRange = {};

		subresRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresRange.levelCount = 1;
		subresRange.layerCount = 1;

		vkCmdClearColorImage(cmdBuffer, geometryPass.headIndex.image, VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &subresRange);

		// Clear previous geometry pass data
		vkCmdFillBuffer(cmdBuffer, geometryPass.geometry.buffer, 0, sizeof(uint32_t), 0);

		// We need a barrier to make sure all writes are finished before starting to write again
		VkMemoryBarrier memoryBarrier = vks::initializers::memoryBarrier();
		memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

		// Begin the geometry render pass
		renderPassBeginInfo.renderPass = geometryPass.renderPass;
		renderPassBeginInfo.framebuffer = geometryPass.framebuffer;
		renderPassBeginInfo.clearValueCount = 0;
		renderPassBeginInfo.pClearValues = nullptr;

		vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.geometry);
		uint32_t dynamicOffset = 0;
		models_.sphere.bindBuffers(cmdBuffer);

		// Render the scene
		ObjectData objectData;

		vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.geometry, 0, 1, &descriptorSets_[currentBuffer_].geometry, 0, nullptr);
		objectData.color = glm::vec4(1.0f, 0.0f, 0.0f, 0.5f);
		for (int32_t x = 0; x < 5; x++)
		{
			for (int32_t y = 0; y < 5; y++)
			{
				for (int32_t z = 0; z < 5; z++)
				{
					glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(x - 2, y - 2, z - 2));
					glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(0.3f));
					objectData.model = T * S;
					vkCmdPushConstants(cmdBuffer, pipelineLayouts_.geometry, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ObjectData), &objectData);
					models_.sphere.draw(cmdBuffer);
				}
			}
		}

		models_.cube.bindBuffers(cmdBuffer);
		objectData.color = glm::vec4(0.0f, 0.0f, 1.0f, 0.5f);
		for (uint32_t x = 0; x < 2; x++)
		{
			glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(3.0f * x - 1.5f, 0.0f, 0.0f));
			glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(0.2f));
			objectData.model = T * S;
			vkCmdPushConstants(cmdBuffer, pipelineLayouts_.geometry, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ObjectData), &objectData);
			models_.cube.draw(cmdBuffer);
		}

		vkCmdEndRenderPass(cmdBuffer);

		// Make a pipeline barrier to guarantee the geometry pass is done
		vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 0, nullptr);

		// We need a barrier to make sure all writes are finished before starting to write again
		memoryBarrier = vks::initializers::memoryBarrier();
		memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

		// Begin the color render pass
		renderPassBeginInfo.renderPass = renderPass_;
		renderPassBeginInfo.framebuffer = frameBuffers_[currentImageIndex_];
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.color);
		vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts_.color, 0, 1, &descriptorSets_[currentBuffer_].color, 0, nullptr);
		vkCmdDraw(cmdBuffer, 3, 1, 0, 0);
		drawUI(cmdBuffer);
		vkCmdEndRenderPass(cmdBuffer);

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

	void windowResized() override
	{
		destroyGeometryPass();
		prepareGeometryPass();
		vkResetDescriptorPool(device_, descriptorPool_, 0);
		updateDescriptors();
		resized_ = false;
	}

	void destroyGeometryPass()
	{
		vkDestroyRenderPass(device_, geometryPass.renderPass, nullptr);
		vkDestroyFramebuffer(device_, geometryPass.framebuffer, nullptr);
		geometryPass.geometry.destroy();
		geometryPass.headIndex.destroy();
		geometryPass.linkedList.destroy();
	}
};

VULKAN_EXAMPLE_MAIN()
