#include "NeuralNetwork\NeuralNet.h"
#include <vector>
#include <iostream>

int main()
{
	// vector of unsigneds representing number of nodes in each layer
	std::vector<size_t> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);

	// create a network
	NeuralNet myNet(topology);

	// data
	std::vector<double> inputs;
	
	for (size_t i = 0; i < 3; i++)
	{
		inputs.push_back(7.0);
	}

	std::vector<double> targetValues;

	for (size_t i = 0; i < 1; i++)
	{
		targetValues.push_back(3);
	}

	std::vector<double> resultValues;

	// feed, train, and get results from the network
	myNet.FeedForward(inputs);
	myNet.BackPropogation(targetValues);
	myNet.GetResults(resultValues);

	char newChar;
	std::cin >> newChar;

	return 0;
}