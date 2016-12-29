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
		0,
		false);
}

void InputLayer::Dispose()
{
	Layer::Dispose();
}

void InputLayer::Forward(double* input, int inputSize)
{
	assert(inputSize == nodeCount);

	memcpy(forwardHostMem.get(), input, nodeCount * sizeof(double));
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