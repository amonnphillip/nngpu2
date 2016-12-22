#pragma once

#include "inputlayerconfig.h"
#include "innetworklayer.h"
#include "layer.h"

class InputNode
{
};

class InputLayer : public Layer<InputNode, double, double, double>, public INNetworkLayer
{
private:
	int nodeCount = 0;

public:
	InputLayer(InputLayerConfig* config, INNetworkLayer* previousLayer);
	virtual void Forward(double* input, int inputSize);
	virtual void Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer);
	virtual void Backward(double* input, int inputSize, double learnRate);
	virtual void Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate);
	virtual void Dispose();
	virtual double* GetForward();
	virtual double* GetBackward();
	virtual int GetForwardNodeCount();
	virtual int GetBackwardNodeCount();
};
