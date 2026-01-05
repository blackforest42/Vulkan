/*
* Vulkan Example - Geometry shader (vertex normal debugging)
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
	bool displayNormals = true;

	vkglTF::Model scene;

	struct UniformData {
		glm::mat4 projection;
		glm::mat4 modelView;
	} uniformData_;
	std::array<vks::Buffer, MAX_CONCURRENT_FRAMES> uniformBuffers_;

	struct {
		VkPipeline solid{ VK_NULL_HANDLE };
		VkPipeline normals{ VK_NULL_HANDLE };
	} pipelines_;

	VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
	VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
	std::array<VkDescriptorSet, MAX_CONCURRENT_FRAMES> descriptorSets_{};

	VulkanExample() : VulkanExampleBase()
	{
		title_ = "Geometry shader normal debugging";
		camera_.type_ = Camera::CameraType::lookat;
		camera_.setPosition(glm::vec3(0.0f, 0.0f, -1.0f));
		camera_.setRotation(glm::vec3(0.0f, -25.0f, 0.0f));
		camera_.setPerspective(60.0f, (float)width_ / (float)height_, 0.1f, 128.0f);
	}

	~VulkanExample()
	{
		if (device_) {
			vkDestroyPipeline(device_, pipelines_.solid, nullptr);
			vkDestroyPipeline(device_, pipelines_.normals, nullptr);
			vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(device_, descriptorSetLayout, nullptr);
			for (auto& buffer : uniformBuffers_) {
				buffer.destroy();
			}
		}
	}

	// Enable physical device features required for this example
	virtual void getEnabledFeatures()
	{
		// Geometry shader support is required for this example
		if (deviceFeatures_.geometryShader) {
			enabledFeatures_.geometryShader = VK_TRUE;
		}
		else {
			vks::tools::exitFatal("Selected GPU does not support geometry shaders!", VK_ERROR_FEATURE_NOT_PRESENT);
		}
	}

	void loadAssets()
	{
		scene.loadFromFile(getAssetPath() + "models/suzanne.gltf", vulkanDevice_, queue_, vkglTF::FileLoadingFlags::PreTransformVertices | vkglTF::FileLoadingFlags::PreMultiplyVertexColors | vkglTF::FileLoadingFlags::FlipY);
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
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0 : Shader uniform ubo
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT, 0),
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
		//Layout
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

		// Pipelines
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_LINE_WIDTH };
		VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables, 0);

		// Tessellation pipeline
		std::array<VkPipelineShaderStageCreateInfo, 3> shaderStages;
		shaderStages[0] = loadShader(getShadersPath() + "geometryshader/base.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "geometryshader/base.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		shaderStages[2] = loadShader(getShadersPath() + "geometryshader/normaldebug.geom.spv", VK_SHADER_STAGE_GEOMETRY_BIT);

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
		pipelineCI.renderPass = renderPass_;
		pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal, vkglTF::VertexComponent::Color });

		// Normal debugging pipeline
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.normals));

		// Solid rendering pipeline
		shaderStages[0] = loadShader(getShadersPath() + "geometryshader/mesh.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "geometryshader/mesh.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		pipelineCI.stageCount = 2;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineCI, nullptr, &pipelines_.solid));
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
		// Vertex shader
		uniformData_.projection = camera_.matrices_.perspective;
		uniformData_.modelView = camera_.matrices_.view;
		memcpy(uniformBuffers_[currentBuffer_].mapped, &uniformData_, sizeof(UniformData));
	}

	void prepare()
	{
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

		VkClearValue clearValues[2]{};
		clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
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

		vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		VkViewport viewport = vks::initializers::viewport((float)width_, (float)height_, 0.0f, 1.0f);
		vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

		VkRect2D scissor = vks::initializers::rect2D(width_, height_, 0, 0);
		vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

		vkCmdSetLineWidth(cmdBuffer, 1.0f);

		vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets_[currentBuffer_], 0, nullptr);

		// Solid shading
		vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.solid);
		scene.draw(cmdBuffer);

		// Normal debugging
		if (displayNormals)
		{
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_.normals);
			scene.draw(cmdBuffer);
		}

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
		if (overlay->header("Settings")) {
			overlay->checkBox("Display normals", &displayNormals);
		}
	}

};

VULKAN_EXAMPLE_MAIN()