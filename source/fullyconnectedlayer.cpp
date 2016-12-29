#include "fullyconnectedlayer.h"
#include "layerexception.h"

extern void FullyConnectedLayer_Forward(FullyConnectedNode *node, double* weights, int weightCount, double *input, double *output, int nodeCount);
extern void FullyConnectedLayer_Backward(FullyConnectedNode *node, double* weights, int weightCount, double *forward, double *previousLayerForward, double* nextlayerBackward, double *output, int nodeCount, double learnRate);

FullyConnectedLayer::FullyConnectedLayer(FullyConnectedLayerConfig* config, INNetworkLayer* previousLayer)
{
	int previousLayerCount = previousLayer->GetForwardNodeCount();
	weightCount = previousLayer->GetForwardNodeCount();
	forwardCount = config->GetWidth() * config->GetHeight() * config->GetDepth();
	nodeCount = forwardCount;
	Layer::Initialize(
		"fullyconnected",
		forwardCount,
		previousLayerCount,
		forwardCount,
		forwardCount,
		true);

	FullyConnectedNode* nodes = nodeHostMem.get();
	for (int index = 0; index < nodeCount; index++)
	{
		nodes->bias = 0;
		nodes++;
	}

	if (cudaMemcpy(nodeDeviceMem, nodeHostMem.get(), nodeCount * sizeof(FullyConnectedNode), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyConnectedLayer cudaMemcpy returned an error");
	}

	weightsHostMem = std::unique_ptr<double>(new double[weightCount * nodeCount]);
	if (weightsHostMem.get() == nullptr)
	{
		throw std::bad_alloc();
	}

	double* weights = weightsHostMem.get();
	for (int index = 0; index < weightCount * nodeCount; index++)
	{
		*weights = 1;
		weights++;
	}

	if (cudaMalloc((void**)&weightsDeviceMem, weightCount * nodeCount * sizeof(double)) != cudaError::cudaSuccess)
	{
		throw std::bad_alloc();
	}

	if (cudaMemcpy(weightsDeviceMem, weightsHostMem.get(), weightCount * nodeCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("FullyConnectedLayer cudaMemcpy returned an error");
	}
}


void FullyConnectedLayer::Dispose()
{

	// TODO: DISPOSE OF WEIGHT MEM

	Layer::Dispose();
}

void FullyConnectedLayer::Forward(double* input, int inputSize)
{
	throw LayerException("Forward variant not valid for Sigmoid layer");
}

void FullyConnectedLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	if (cudaMemcpy(inputDeviceMem, previousLayer->GetForwardHostMem(), nodeCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid forward cudaMemcpy returned an error");
	}

	FullyConnectedLayer_Forward(nodeDeviceMem, weightsDeviceMem, weightCount, inputDeviceMem, forwardDeviceMem, nodeCount);

	if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, nodeCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid forward cudaMemcpy returned an error");
	}
}

void FullyConnectedLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for Sigmoid layer");
}

void FullyConnectedLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	double* backward = backwardHostMem.get();
	for (int index = 0; index < backwardCount; index++)
	{
		*backward = 0;
		backward++;
	}

	if (cudaMemcpy(backwardDeviceMem, backwardHostMem.get(), backwardCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid backward cudaMemcpy returned an error");
	}

	// TODO: NOT REALLY NEEDED! CAN JUST USE DEVICE MEMORY INSTEAD
	if (cudaMemcpy(inputDeviceMem, previousLayer->GetForwardHostMem(), nodeCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid backward cudaMemcpy returned an error");
	}

	FullyConnectedLayer_Backward(nodeDeviceMem, weightsDeviceMem, weightCount, forwardDeviceMem, inputDeviceMem, nextLayer->GetBackwardDeviceMem(), backwardDeviceMem, nodeCount, learnRate);

	if (cudaMemcpy(weightsHostMem.get(), weightsDeviceMem, weightCount * nodeCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid backward cudaMemcpy returned an error");
	}

	if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, nodeCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid backward cudaMemcpy returned an error");
	}
}

double* FullyConnectedLayer::GetForwardHostMem()
{
	return forwardHostMem.get();
}

double* FullyConnectedLayer::GetBackwardHostMem()
{
	return backwardHostMem.get();
}

double* FullyConnectedLayer::GetForwardDeviceMem()
{
	return forwardDeviceMem;
}

double* FullyConnectedLayer::GetBackwardDeviceMem()
{
	return backwardDeviceMem;
}

int FullyConnectedLayer::GetForwardNodeCount()
{
	return nodeCount;
}

int FullyConnectedLayer::GetBackwardNodeCount()
{
	return nodeCount;
}

double* FullyConnectedLayer::GetWeightsForNode(int index)
{
	// TODO: assert for out of bounds index

	return weightsHostMem.get() + (weightCount * index);
}