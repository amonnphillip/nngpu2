#include "inputlayerconfig.h"
#include "inputlayer.h"
#include "layerexception.h"
#include <cassert>
#include "cuda_runtime.h"
#include "layer.h"

InputLayer::InputLayer(InputLayerConfig* config, INNetworkLayer* previousLayer)
{
	nodeCount = config->GetWidth() * config->GetHeight() * config->GetDepth();
	Layer::Initialize(
		"input",
		nodeCount,
		0,
		0,
		true);
}

void InputLayer::Dispose()
{
	Layer::Dispose();
}

void InputLayer::Forward(double* input, int inputSize)
{
	assert(inputSize == nodeCount);

	// TODO: Maybe we dont need to copy here?
	memcpy(forwardHostMem.get(), input, nodeCount * sizeof(double));

	if (cudaMemcpy(forwardDeviceMem, forwardHostMem.get(), nodeCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("InputLayer forward cudaMemcpy returned an error");
	}
}

void InputLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	throw LayerException("Forward variant not valid for InputLayer layer");
}

void InputLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for InputLayer layer");
}

void InputLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	throw LayerException("Backward variant not valid for InputLayer layer");
}

double* InputLayer::GetForwardHostMem()
{
	return forwardHostMem.get();
}

double* InputLayer::GetBackwardHostMem()
{
	return nullptr;
}

double* InputLayer::GetForwardDeviceMem()
{
	return forwardDeviceMem;
}

double* InputLayer::GetBackwardDeviceMem()
{
	return nullptr;
}

int InputLayer::GetForwardNodeCount()
{
	return nodeCount;
}

int InputLayer::GetBackwardNodeCount()
{
	return 0;
}

int InputLayer::GetWidth()
{
	return nodeCount;
}

int InputLayer::GetHeight()
{
	return 1;
}

int InputLayer::GetDepth()
{
	return 1;
}

std::string InputLayer::GetLayerName()
{
	return Layer::GetLayerName();
}