//////////////////////////////////////////////////////
// 
// File: NeuralNet.cpp
// Purpose: Contains the Neural Network Class
// 
// Author: Brett Schiff
// Contact: brettschiff@gmail.com
// 
//////////////////////////////////////////////////////
#include "NeuralNet.h"
#include "Neuron.h"
#include <iostream>
#include <cassert>

// constructors
NeuralNet::NeuralNet(const std::vector<size_t> &topology)
{
	// the number of layers
	size_t numLayers = topology.size();

	// add each layer to the neural net
	for (size_t i = 0; i < numLayers; ++i)
	{
		//!?!? TEMP
		std::cout << "starting layer: " << i << "\n";

		// get the number of neurons in the next layer, if there is a next layer
		size_t numOutputs = (i == (numLayers - 1)) ? 0 : topology[i + 1];

		// add a new layer
		m_layers.push_back(Layer());

		// add the neurons to that layer(one extra for biases)
		for (size_t j = 0; j <= topology[i]; j++)
		{
			//!?!? TEMP
			std::cout << "created a neuron\n";

			m_layers.back().push_back(Neuron(numOutputs, j));
		}

		// set the bias neuron's output to 1
		m_layers.back().back().SetOutputValue(1.0);

		//!?!? TEMP
		std::cout << "finished layer " << i << " with " << topology[i] << " neurons\n\n";
	}
}

// methods
void NeuralNet::FeedForward(const std::vector<double> &inputs)
{
	// assert that we have the correct number of inputs
	assert(inputs.size() == NumberOfNeuronsInLayer(0));

	// assign the input values to the input neurons
	for (size_t i = 0; i < NumberOfNeuronsInLayer(0); ++i)
	{
		m_layers[0][i].SetOutputValue(inputs[i]);
	}

	// Forward Propogate: go through each layer after the input layer
	for (size_t i = 1; i < NumberOfLayers(); ++i)
	{
		Layer& previousLayer = m_layers[i - 1];

		// go through each neuron in each layer
		for (size_t j = 0; j < NumberOfNeuronsInLayer(i); ++j)
		{
			// have the neurons update from the previous layer of neurons
			m_layers[i][j].FeedForward(previousLayer);
		}
	}
}

void NeuralNet::BackPropogation(const std::vector<double> &targetValues)
{
	// assert that we are given an appopriate amount of target values
	assert(targetValues.size() == NumberOfNeuronsInLayer(m_layers.back()));

	// Calculate overall error(Root Mean Square)
	Layer& outputLayer = m_layers.back();
	m_error = 0;

	size_t numOutputNeurons = NumberOfNeuronsInLayer(outputLayer);

	for (size_t i = 0; i < numOutputNeurons; ++i)
	{
		// difference between expected and desired result
		double difference = targetValues[i] - outputLayer[i].GetOutputValue();
		// sum them together
		m_error += difference * difference;
	}

	m_error /= numOutputNeurons; // take the average
	m_error = sqrt(m_error); // defined by RMS(Root Mean Square)

	// Calculate output layer gradients
	for (size_t i = 0; i < numOutputNeurons; ++i)
	{
		outputLayer[i].CalculateOutputGradients(targetValues[i]);
	}
	
	// Calculate gradients on hidden layers
	for (size_t i = NumberOfLayers() - 2; i > 0; --i)
	{
		Layer& hiddenLayer = m_layers[i];
		Layer& nextLayer = m_layers[i + 1];

		// include the bias neuron
		for (size_t i = 0; i < hiddenLayer.size(); ++i)
		{
			hiddenLayer[i].CalculateHiddenGradients(nextLayer);
		}
	}
	
	// Update connection weights on all layers
	// go through each layer
	for (size_t i = NumberOfLayers() - 1; i > 0; --i)
	{
		Layer& currentLayer = m_layers[i];
		Layer& previousLayer = m_layers[i - 1];

		for (size_t j = 0; j < NumberOfNeuronsInLayer(currentLayer); ++j)
		{
			currentLayer[j].UpdateInputWeights(previousLayer);
		}
	}
}

void NeuralNet::GetResults(std::vector<double> &resultValues) const
{
	// empty out the container
	resultValues.clear();

	const Layer& outputLayer = m_layers.back();
	size_t numberOfOutputNeurons = NumberOfNeuronsInLayer(outputLayer);

	// go through each output neuron
	for (size_t i = 0; i < numberOfOutputNeurons; ++i)
	{
		// and push them onto the results
		resultValues.push_back(outputLayer[i].GetOutputValue());
	}
}

// private methods
size_t NeuralNet::NumberOfLayers() const
{
	return m_layers.size();
}

size_t NeuralNet::NumberOfNeuronsInLayer(size_t	layerNumber) const
{
	// one less than size to account for the hidden bias neuron
	return m_layers[layerNumber].size() - 1;
}

size_t NeuralNet::NumberOfNeuronsInLayer(const Layer& layer)
{
	// one less than size to account for the hidden bias neuron
	return layer.size() - 1;
}
