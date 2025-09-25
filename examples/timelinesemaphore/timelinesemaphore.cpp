/*
* Vulkan Example - Using timeline semaphores
* 
* Based on the compute n-nbody sample, this sample replaces multiple semaphores with a single timeline semaphore
*
* Copyright (C) 2024-2025 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"

#if defined(__ANDROID__)
// Lower particle count on Android for performance reasons
constexpr auto  PARTICLES_PER_ATTRACTOR = 3 * 1024;
#else
constexpr auto PARTICLES_PER_ATTRACTOR = 4 * 1024;
#endif

class VulkanExample : public VulkanExampleBase
{
public:
	struct Textures {
		vks::Texture2D particle;
		vks::Texture2D gradient;
	} textures{};

	// Particle Definition
	struct Particle {
		glm::vec4 pos;
		glm::vec4 vel;
	};
	uint32_t numParticles{ 0 };
	vks::Buffer storageBuffer;

	// Resources for the graphics part of the example
	struct Graphics {
		uint32_t queueFamilyIndex;
		VkDescriptorSetLayout descriptorSetLayout;
		std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSets;
		VkPipelineLayout pipelineLayout;
		VkPipeline pipeline;
		struct UniformData {
			glm::mat4 projection;
			glm::mat4 view;
			glm::vec2 screenDim;
		} uniformData;
		std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers;
	} graphics{};

	// Resources for the compute part of the example
	struct Compute {
		uint32_t queueFamilyIndex;
		VkQueue queue;
		VkCommandPool commandPool;
		std::array<VkCommandBuffer, MAX_CONCURRENT_FRAMES> commandBuffers;
		VkDescriptorSetLayout descriptorSetLayout;
		std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSets;
		VkPipelineLayout pipelineLayout;
		VkPipeline pipelineCalculate;
		VkPipeline pipelineIntegrate;
		struct UniformData {
			float deltaT{ 0.0f };
			int32_t particleCount{ 0 };
			float gravity{ 0.002f };
			float power{ 0.75f };
			float soften{ 0.05f };
		} uniformData;
		std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers;
	} compute{};

	// Along with the actual semaphore we also need to track the increasing value of the timeline, 
	// so we store both in a single struct
	struct TimeLineSemaphore {
		VkSemaphore handle{ VK_NULL_HANDLE };
		uint64_t value{ 0 };
	} timeLineSemaphore;

	VkPhysicalDeviceTimelineSemaphoreFeaturesKHR enabledTimelineSemaphoreFeaturesKHR{};
	PFN_vkWaitSemaphoresKHR vkWaitSemaphoresKHR{ VK_NULL_HANDLE };

	VulkanExample() : VulkanExampleBase()
	{
		title = "Timeline semaphores";
		camera_.type = Camera::CameraType::lookat;
		camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 512.0f);
		camera_.setRotation(glm::vec3(-26.0f, 75.0f, 0.0f));
		camera_.setTranslation(glm::vec3(0.0f, 0.0f, -14.0f));
		camera_.movementSpeed = 2.5f;

		enabledInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);

		enabledTimelineSemaphoreFeaturesKHR.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR;
		enabledTimelineSemaphoreFeaturesKHR.timelineSemaphore = VK_TRUE;

		deviceCreatepNextChain = &enabledTimelineSemaphoreFeaturesKHR;
	}

	~VulkanExample()
	{
		if (device_) {
			vkDestroySemaphore(device_, timeLineSemaphore.handle, nullptr);

			// Graphics
			vkDestroyPipeline(device_, graphics.pipeline, nullptr);
			vkDestroyPipelineLayout(device_, graphics.pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(device_, graphics.descriptorSetLayout, nullptr);
			for (auto& buffer : graphics.uniformBuffers) {
				buffer.destroy();
			}

			// Compute
			vkDestroyPipelineLayout(device_, compute.pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(device_, compute.descriptorSetLayout, nullptr);
			vkDestroyPipeline(device_, compute.pipelineCalculate, nullptr);
			vkDestroyPipeline(device_, compute.pipelineIntegrate, nullptr);
			vkDestroyCommandPool(device_, compute.commandPool, nullptr);
			for (auto& buffer : compute.uniformBuffers) {
				buffer.destroy();
			}

			storageBuffer.destroy();

			textures.particle.destroy();
			textures.gradient.destroy();
		}
	}

	void loadAssets()
	{
		textures.particle.loadFromFile(getAssetPath() + "textures/particle01_rgba.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue_);
		textures.gradient.loadFromFile(getAssetPath() + "textures/particle_gradient_rgba.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue_);
	}

	// Setup and fill the compute shader storage buffers containing the particles
	void prepareStorageBuffers()
	{
		// We mark a few particles as attractors that move along a given path, these will pull in the other particles
		std::vector<glm::vec3> attractors = {
			glm::vec3(5.0f, 0.0f, 0.0f),
			glm::vec3(-5.0f, 0.0f, 0.0f),
			glm::vec3(0.0f, 0.0f, 5.0f),
			glm::vec3(0.0f, 0.0f, -5.0f),
			glm::vec3(0.0f, 4.0f, 0.0f),
			glm::vec3(0.0f, -8.0f, 0.0f),
		};

		numParticles = static_cast<uint32_t>(attractors.size()) * PARTICLES_PER_ATTRACTOR;

		// Initial particle positions
		std::vector<Particle> particleBuffer(numParticles);

		std::default_random_engine rndEngine(benchmark.active ? 0 : (unsigned)time(nullptr));
		std::normal_distribution<float> rndDist(0.0f, 1.0f);

		for (uint32_t i = 0; i < static_cast<uint32_t>(attractors.size()); i++) {
			for (uint32_t j = 0; j < PARTICLES_PER_ATTRACTOR; j++) {
				Particle& particle = particleBuffer[i * PARTICLES_PER_ATTRACTOR + j];
				// First particle in group as heavy center of gravity
				if (j == 0)
				{
					particle.pos = glm::vec4(attractors[i] * 1.5f, 90000.0f);
					particle.vel = glm::vec4(glm::vec4(0.0f));
				}
				else
				{
					// Position
					glm::vec3 position(attractors[i] + glm::vec3(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine)) * 0.75f);
					float len = glm::length(glm::normalize(position - attractors[i]));
					position.y *= 2.0f - (len * len);

					// Velocity
					glm::vec3 angular = glm::vec3(0.5f, 1.5f, 0.5f) * (((i % 2) == 0) ? 1.0f : -1.0f);
					glm::vec3 velocity = glm::cross((position - attractors[i]), angular) + glm::vec3(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine) * 0.025f);

					float mass = (rndDist(rndEngine) * 0.5f + 0.5f) * 75.0f;
					particle.pos = glm::vec4(position, mass);
					particle.vel = glm::vec4(velocity, 0.0f);
				}
				// Color gradient offset
				particle.vel.w = (float)i * 1.0f / static_cast<uint32_t>(attractors.size());
			}
		}

		compute.uniformData.particleCount = numParticles;

		VkDeviceSize storageBufferSize = particleBuffer.size() * sizeof(Particle);

		// Staging
		vks::Buffer stagingBuffer;
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer, storageBufferSize, particleBuffer.data());
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &storageBuffer, storageBufferSize);

		// Copy from staging buffer to storage buffer
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = storageBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, storageBuffer.buffer, 1, &copyRegion);
		// Execute a transfer barrier to the compute queue, if necessary
		if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
		{
			VkBufferMemoryBarrier buffer_barrier =
			{
				VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
				nullptr,
				VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
				0,
				graphics.queueFamilyIndex,
				compute.queueFamilyIndex,
				storageBuffer.buffer,
				0,
				storageBuffer.size
			};

			vkCmdPipelineBarrier(
				copyCmd,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
				VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
				0,
				0, nullptr,
				1, &buffer_barrier,
				0, nullptr);
		}
		vulkanDevice->flushCommandBuffer(copyCmd, queue_, true);

		stagingBuffer.destroy();
	}

	void prepareDescriptorPool()
	{
		// This is shared between graphics and compute
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_CONCURRENT_FRAMES * 2),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_CONCURRENT_FRAMES),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_CONCURRENT_FRAMES * 2)
		};
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, MAX_CONCURRENT_FRAMES * 2);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolInfo, nullptr, &descriptorPool_));
	}

	void prepareGraphics()
	{
		// Descriptor layout
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 2),
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorLayout, nullptr, &graphics.descriptorSetLayout));

		// Vertex shader uniform buffer block
		for (auto& buffer : graphics.uniformBuffers) {
			vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer, sizeof(Graphics::UniformData));
			VK_CHECK_RESULT(buffer.map());
		}

		// Sets per frame, just like the buffers themselves
		for (auto i = 0; i < graphics.uniformBuffers.size(); i++) {
			VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool_, &graphics.descriptorSetLayout, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo, &graphics.descriptorSets[i]));
			std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
			   vks::initializers::writeDescriptorSet(graphics.descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &textures.particle.descriptor),
			   vks::initializers::writeDescriptorSet(graphics.descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &textures.gradient.descriptor),
			   vks::initializers::writeDescriptorSet(graphics.descriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2, &graphics.uniformBuffers[i].descriptor),
			};
			vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
		}

		// Pipeline layout
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&graphics.descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr, &graphics.pipelineLayout));

		// Pipeline
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_ALWAYS);
		VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

		// Vertex Input state
		std::vector<VkVertexInputBindingDescription> inputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(Particle), VK_VERTEX_INPUT_RATE_VERTEX)
		};
		std::vector<VkVertexInputAttributeDescription> inputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Particle, pos)),
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Particle, vel)),
		};
		VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(inputBindings.size());
		vertexInputState.pVertexBindingDescriptions = inputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(inputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = inputAttributes.data();

		// Shaders
		shaderStages[0] = loadShader(getShadersPath() + "computenbody/particle.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "computenbody/particle.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

		VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(graphics.pipelineLayout, renderPass_, 0);
		pipelineCreateInfo.pVertexInputState = &vertexInputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();
		pipelineCreateInfo.renderPass = renderPass_;

		// Additive blending
		blendAttachmentState.colorWriteMask = 0xF;
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache, 1, &pipelineCreateInfo, nullptr, &graphics.pipeline));
	}

	void prepareCompute()
	{
		vkGetDeviceQueue(device_, compute.queueFamilyIndex, 0, &compute.queue);
		for (auto& buffer : compute.uniformBuffers) {
			vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer, sizeof(Compute::UniformData));
			VK_CHECK_RESULT(buffer.map());
		}
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0 : Particle position storage buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
			// Binding 1 : Uniform buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &descriptorLayout, nullptr, &compute.descriptorSetLayout));
		for (auto i = 0; i < compute.uniformBuffers.size(); i++) {
			VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool_, &compute.descriptorSetLayout, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocInfo, &compute.descriptorSets[i]));
			std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
				vks::initializers::writeDescriptorSet(compute.descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &storageBuffer.descriptor),
				vks::initializers::writeDescriptorSet(compute.descriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1, &compute.uniformBuffers[i].descriptor)
			};
			vkUpdateDescriptorSets(device_, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, nullptr);
		}
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout));
		VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);
		computePipelineCreateInfo.stage = loadShader(getShadersPath() + "computenbody/particle_calculate.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
		uint32_t sharedDataSize = std::min((uint32_t)1024, (uint32_t)(vulkanDevice->properties.limits.maxComputeSharedMemorySize / sizeof(glm::vec4)));
		VkSpecializationMapEntry specializationMapEntry = vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
		VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(1, &specializationMapEntry, sizeof(int32_t), &sharedDataSize);
		computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;
		VK_CHECK_RESULT(vkCreateComputePipelines(device_, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipelineCalculate));
		computePipelineCreateInfo.stage = loadShader(getShadersPath() + "computenbody/particle_integrate.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
		VK_CHECK_RESULT(vkCreateComputePipelines(device_, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipelineIntegrate));
		
		VkCommandPoolCreateInfo cmdPoolInfo = vks::initializers::commandPoolCreateInfo();
		cmdPoolInfo.queueFamilyIndex = compute.queueFamilyIndex;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device_, &cmdPoolInfo, nullptr, &compute.commandPool));
		for (auto& cmdBuffer : compute.commandBuffers) {
			cmdBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, compute.commandPool);
		}
	}

	void updateComputeUniformBuffers()
	{
		compute.uniformData.deltaT = paused ? 0.0f : frameTimer * 0.05f;
		memcpy(compute.uniformBuffers[currentBuffer_].mapped, &compute.uniformData, sizeof(Compute::UniformData));
	}

	void updateGraphicsUniformBuffers()
	{
		graphics.uniformData.projection = camera_.matrices.perspective;
		graphics.uniformData.view = camera_.matrices.view;
		graphics.uniformData.screenDim = glm::vec2((float)width_, (float)height_);
		memcpy(graphics.uniformBuffers[currentBuffer_].mapped, &graphics.uniformData, sizeof(Graphics::UniformData));
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		graphics.queueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
		compute.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;

		// Setup the timeline semaphore
		VkSemaphoreCreateInfo semaphoreCI{};
		semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		// It's a variation of the core semaphore type, creation is handled via an extension struture
		VkSemaphoreTypeCreateInfoKHR semaphoreTypeCI{};
		semaphoreTypeCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR;
		semaphoreTypeCI.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR;
		semaphoreTypeCI.initialValue = timeLineSemaphore.value;

		semaphoreCI.pNext = &semaphoreTypeCI;
		VK_CHECK_RESULT(vkCreateSemaphore(device_, &semaphoreCI, nullptr, &timeLineSemaphore.handle));

		// We need this extension function to use the timeline semaphore as a replacement for fences
		vkWaitSemaphoresKHR = reinterpret_cast<PFN_vkWaitSemaphoresKHR>(vkGetDeviceProcAddr(device_, "vkWaitSemaphoresKHR"));

		loadAssets();
		prepareDescriptorPool();
		prepareStorageBuffers();
		prepareGraphics();
		prepareCompute();
		prepared_ = true;
	}

	void buildGraphicsCommandBuffer()
	{
		VkCommandBuffer cmdBuffer = drawCmdBuffers[currentBuffer_];
		
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2]{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass_;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width_;
		renderPassBeginInfo.renderArea.extent.height = height_;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;
		renderPassBeginInfo.framebuffer = frameBuffers_[currentImageIndex_];

		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

		// Acquire barrier
		if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
		{
			VkBufferMemoryBarrier buffer_barrier =
			{
				VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
				nullptr,
				0,
				VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
				compute.queueFamilyIndex,
				graphics.queueFamilyIndex,
				storageBuffer.buffer,
				0,
				storageBuffer.size
			};

			vkCmdPipelineBarrier(
				cmdBuffer,
				VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
				0,
				0, nullptr,
				1, &buffer_barrier,
				0, nullptr);
		}

		// Draw the particle system using the update vertex buffer
		vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		VkViewport viewport = vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
		vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

		VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
		vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

		vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipeline);
		vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout, 0, 1, &graphics.descriptorSets[currentBuffer_], 0, nullptr);

		VkDeviceSize offsets[1] = { 0 };
		vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &storageBuffer.buffer, offsets);
		vkCmdDraw(cmdBuffer, numParticles, 1, 0, 0);

		drawUI(cmdBuffer);

		vkCmdEndRenderPass(cmdBuffer);

		// Release barrier
		if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
		{
			VkBufferMemoryBarrier buffer_barrier =
			{
				VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
				nullptr,
				VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
				0,
				graphics.queueFamilyIndex,
				compute.queueFamilyIndex,
				storageBuffer.buffer,
				0,
				storageBuffer.size
			};

			vkCmdPipelineBarrier(
				cmdBuffer,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
				VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
				0,
				0, nullptr,
				1, &buffer_barrier,
				0, nullptr);
		}

		VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
	}

	void buildComputeCommandBuffer()
	{
		VkCommandBuffer cmdBuffer = compute.commandBuffers[currentBuffer_];
		
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

		// Acquire barrier
		if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
		{
			VkBufferMemoryBarrier bufferBarrier =
			{
				VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
				nullptr,
				0,
				VK_ACCESS_SHADER_WRITE_BIT,
				graphics.queueFamilyIndex,
				compute.queueFamilyIndex,
				storageBuffer.buffer,
				0,
				storageBuffer.size
			};

			vkCmdPipelineBarrier(
				cmdBuffer,
				VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0,
				0, nullptr,
				1, &bufferBarrier,
				0, nullptr);
		}

		// First pass: Calculate particle movement
		// -------------------------------------------------------------------------------------------------------
		vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineCalculate);
		vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSets[currentBuffer_], 0, 0);
		vkCmdDispatch(cmdBuffer, numParticles / 256, 1, 1);

		// Add memory barrier to ensure that the computer shader has finished writing to the buffer
		{
			VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
			bufferBarrier.buffer = storageBuffer.buffer;
			bufferBarrier.size = storageBuffer.descriptor.range;
			bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			// Transfer ownership if compute and graphics queue family indices differ
			bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				cmdBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_FLAGS_NONE,
				0, nullptr,
				1, &bufferBarrier,
				0, nullptr);
		}

		// Second pass: Integrate particles
		// -------------------------------------------------------------------------------------------------------
		vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineIntegrate);
		vkCmdDispatch(cmdBuffer, numParticles / 256, 1, 1);

		// Release barrier
		if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
		{
			VkBufferMemoryBarrier bufferBarrier =
			{
				VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
				nullptr,
				VK_ACCESS_SHADER_WRITE_BIT,
				0,
				compute.queueFamilyIndex,
				graphics.queueFamilyIndex,
				storageBuffer.buffer,
				0,
				storageBuffer.size
			};

			vkCmdPipelineBarrier(
				cmdBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
				0,
				0, nullptr,
				1, &bufferBarrier,
				0, nullptr);
		}

		vkEndCommandBuffer(cmdBuffer);
	}


	virtual void render()
	{
		if (!prepared_)
			return;

		// Define incremental timeline sempahore states used to wait and signal
		const uint64_t graphicsFinished = timeLineSemaphore.value;
		const uint64_t computeFinished = timeLineSemaphore.value + 1;
		const uint64_t allFinished = timeLineSemaphore.value + 2;

		VulkanExampleBase::prepareFrame(false);
		
		// Timeline semaphores can also be used to replace fences
		VkSemaphoreWaitInfo semaphoreWaitInfo{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
		semaphoreWaitInfo.semaphoreCount = 1;
		semaphoreWaitInfo.pSemaphores = &timeLineSemaphore.handle;
		semaphoreWaitInfo.pValues = &timeLineSemaphore.value;
		vkWaitSemaphoresKHR(device_, &semaphoreWaitInfo, UINT64_MAX);

		updateComputeUniformBuffers();
		buildComputeCommandBuffer();

		// Wait for rendering finished
		VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

		// Submit compute commands

		// With timeline semaphores, we can state on what value we want to wait on / signal on
		VkTimelineSemaphoreSubmitInfoKHR timeLineSubmitInfo{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR };
		timeLineSubmitInfo.waitSemaphoreValueCount = 1;
		timeLineSubmitInfo.pWaitSemaphoreValues = &graphicsFinished;
		timeLineSubmitInfo.signalSemaphoreValueCount = 1;
		timeLineSubmitInfo.pSignalSemaphoreValues = &computeFinished;

		VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &compute.commandBuffers[currentBuffer_];
		computeSubmitInfo.waitSemaphoreCount = 1;
		computeSubmitInfo.pWaitSemaphores = &timeLineSemaphore.handle;
		computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
		computeSubmitInfo.signalSemaphoreCount = 1;
		computeSubmitInfo.pSignalSemaphores = &timeLineSemaphore.handle;
		computeSubmitInfo.pNext = &timeLineSubmitInfo;

		VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE));

		updateGraphicsUniformBuffers();
		buildGraphicsCommandBuffer();

		VkPipelineStageFlags graphicsWaitStageMasks[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		VkSemaphore graphicsWaitSemaphores[] = { timeLineSemaphore.handle, presentCompleteSemaphores[currentBuffer_] };
		VkSemaphore graphicsSignalSemaphores[] = { timeLineSemaphore.handle, renderCompleteSemaphores[currentImageIndex_] };

		// Submit graphics commands
		VkSubmitInfo submitInfo = vks::initializers::submitInfo();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer_];
		submitInfo.waitSemaphoreCount = 2;
		submitInfo.pWaitSemaphores = graphicsWaitSemaphores;
		submitInfo.pWaitDstStageMask = graphicsWaitStageMasks;
		submitInfo.signalSemaphoreCount = 2;
		submitInfo.pSignalSemaphores = graphicsSignalSemaphores;

		uint64_t wait_values[2] = { computeFinished, computeFinished };
		uint64_t signal_values[2] = { allFinished, allFinished };

		timeLineSubmitInfo.waitSemaphoreValueCount = 2;
		timeLineSubmitInfo.pWaitSemaphoreValues = &wait_values[0];
		timeLineSubmitInfo.signalSemaphoreValueCount = 2;
		timeLineSubmitInfo.pSignalSemaphoreValues = &signal_values[0];

		submitInfo.pNext = &timeLineSubmitInfo;

		VK_CHECK_RESULT(vkQueueSubmit(queue_, 1, &submitInfo, VK_NULL_HANDLE));

		// Increase timeline value base for next frame
		timeLineSemaphore.value = allFinished;

		VulkanExampleBase::submitFrame(true);
	}
};

VULKAN_EXAMPLE_MAIN()
