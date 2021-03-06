#include "relulayer.h"
#include "layerexception.h"
#include <cassert>

extern void ReluLayer_Forward(ReluNode *node, double *previousLayerForward, double *output, int nodeCount);
extern void ReluLayer_Backward(ReluNode *node, double *forward, double* nextlayerBackward, double *output, int nodeCount, double learnRate);

ReluLayer::ReluLayer(ReluLayerConfig* config, INNetworkLayer* previousLayer)
{
	backwardCount = previousLayer->GetForwardNodeCount();
	forwardCount = config->GetWidth() * config->GetHeight() * config->GetDepth();
	nodeCount = forwardCount;

	layerWidth = config->GetWidth();
	layerHeight = config->GetHeight();
	layerDepth = config->GetDepth();

	Layer::Initialize(
		"relu",
		forwardCount,
		backwardCount,
		nodeCount,
		true);
}


void ReluLayer::Dispose()
{
	Layer::Dispose();
}

void ReluLayer::Forward(double* input, int inputSize)
{
	throw LayerException("Forward variant not valid for Sigmoid layer");
}

void ReluLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	ReluLayer_Forward(nodeDeviceMem, previousLayer->GetForwardDeviceMem(), forwardDeviceMem, nodeCount);

	if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, forwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("ReluLayer forward cudaMemcpy returned an error");
	}
}

void ReluLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for Sigmoid layer");
}

void ReluLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	double* backward = backwardHostMem.get();
	for (int index = 0; index < backwardCount; index++)
	{
		*backward = 0;
		backward++;
	}

	if (cudaMemcpy(backwardDeviceMem, backwardHostMem.get(), backwardCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyConnectedLayer backward cudaMemcpy returned an error");
	}

	ReluLayer_Backward(nodeDeviceMem, forwardDeviceMem, nextLayer->GetBackwardDeviceMem(), backwardDeviceMem, nodeCount, learnRate);

	if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, backwardCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyConnectedLayer backward cudaMemcpy returned an error");
	}
}

double* ReluLayer::GetForwardHostMem()
{
	return forwardHostMem.get();
}

double* ReluLayer::GetBackwardHostMem()
{
	return backwardHostMem.get();
}

double* ReluLayer::GetForwardDeviceMem()
{
	return forwardDeviceMem;
}

double* ReluLayer::GetBackwardDeviceMem()
{
	return backwardDeviceMem;
}

int ReluLayer::GetForwardNodeCount()
{
	return forwardCount;
}

int ReluLayer::GetBackwardNodeCount()
{
	return backwardCount;
}

int ReluLayer::GetWidth()
{
	return layerWidth;
}

int ReluLayer::GetHeight()
{
	return layerHeight;
}

int ReluLayer::GetDepth()
{
	return layerDepth;
}

std::string ReluLayer::GetLayerName()
{
	return Layer::GetLayerName();
}