//////////////////////////////////////////////////////
// 
// File: NeuralNet.h
// Purpose: Contains the Neural Network Class
// 
// Author: Brett Schiff
// Contact: brettschiff@gmail.com
// 
//////////////////////////////////////////////////////
#pragma once
#include <vector>
#include "Neuron.h"

typedef std::vector<Neuron> Layer;  // typedef of a layer of neurons

class NeuralNet
{
public:
	// constructors
	NeuralNet(const std::vector<size_t> &topology);

	// methods
	void FeedForward(const std::vector<double> &inputs);
	void BackPropogation(const std::vector<double> &targetValues);
	void GetResults(std::vector<double> &resultValues) const;

private:
	// private methods
	size_t NumberOfLayers() const;
	size_t NumberOfNeuronsInLayer(size_t layerNumber) const;
	static size_t NumberOfNeuronsInLayer(const Layer& layer);

	double m_error;              // error of the current net
	std::vector<Layer> m_layers; // layers of neurons
};

