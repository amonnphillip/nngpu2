#pragma once

#include "innetworklayer.h"
#include "layer.h"

class FullyConnectedNode
{
	static const int maxWeights = 100;
	int weightCount = 0;
	double weights[maxWeights];
};

class FullyConnectedLayer : public Layer<FullyConnectedNode, double, double, double>, public INNetworkLayer
{
private:
	int nodeCount = 0;
	int forwardCount = 0;
	int backwardCount = 0;
	int inputCount = 0;

public:
	FullyConnectedLayer(int dimentionx, int dimentiony, int dimentionz, int forwardx, int forwardy, int forwardz, INNetworkLayer* previousLayer);

};

