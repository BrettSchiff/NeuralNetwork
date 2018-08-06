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
#include <fstream>

// constructors
NeuralNet::NeuralNet(const std::vector<size_t> &topology)
{
	// the number of layers
	size_t numLayers = topology.size();

	// add each layer to the neural net
	for (size_t i = 0; i < numLayers; ++i)
	{
		// get the number of neurons in the next layer, if there is a next layer, otherwise 0
		size_t numOutputs = (i == (numLayers - 1)) ? 0 : topology[i + 1];

		// add a new layer
		m_layers.push_back(Layer());

		// add the neurons to that layer(one extra for biases)
		for (size_t j = 0; j <= topology[i]; j++)
		{
			m_layers.back().push_back(Neuron(numOutputs, j));
		}

		// set the bias neuron's output to 1
		m_layers.back().back().SetOutputValue(1.0);
	}
}

NeuralNet::NeuralNet(const std::vector<size_t>& topology, const std::vector<float>& weights)
{
	SetupFromVectors(topology, weights);
}

NeuralNet::NeuralNet(std::string filename)
{
	std::ifstream infile(filename);
	assert(infile.is_open());

	SerializedNeuralNet loadedVector;

	size_t topographySize;
	infile >> topographySize;
	for (size_t i = 0; i < topographySize; i++)
	{
		size_t nextSize;
		infile >> nextSize;
		loadedVector.first.push_back(nextSize);
	}

	size_t weightSize;
	infile >> weightSize;
	for (size_t i = 0; i < weightSize; i++)
	{
		float nextWeight;
		infile >> nextWeight;
		loadedVector.second.push_back(nextWeight);
	}

	SetupFromVectors(loadedVector.first, loadedVector.second);
}

// methods
void NeuralNet::FeedForward(const std::vector<float> &inputs)
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

void NeuralNet::BackPropogation(const std::vector<float> &targetValues)
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
		float difference = targetValues[i] - outputLayer[i].GetOutputValue();
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

void NeuralNet::GetResults(std::vector<float> &resultValues) const
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

void NeuralNet::SerializeToVector(SerializedNeuralNet& serializedNet) const
{
	// record the topological data
	size_t numSerializableLayers = NumberOfLayers();
	for (size_t i = 0; i < numSerializableLayers; ++i)
	{
		serializedNet.first.push_back(m_layers[i].size());
	}
	// record all of the weights(-1 because the last layer has no weights)
	for (size_t i = 0; i < numSerializableLayers - 1; ++i)
	{
		size_t neuronsInCurrLayer = m_layers[i].size();
		for (size_t j = 0; j < neuronsInCurrLayer; ++j)
		{
			m_layers[i][j].StoreWeights_Concat(serializedNet.second);
		}
	}
}

void NeuralNet::GeneticBlend(SerializedNeuralNet& const parent1, SerializedNeuralNet& const parent2, SerializedNeuralNet& child)
{
	assert(parent1.first.size() == parent2.first.size());
	assert(parent1.second.size() == parent2.second.size());

	// make sure the child is empty
	child.first.clear();
	child.second.clear();

	// copy the topography
	child.first = parent1.first;

	// blend the weights
	for (size_t i = 0; i < parent1.second.size(); ++i)
	{
		float blend = parent1.second[i] + ((parent2.second[i] - parent1.second[i]) * RandomScalar());
		child.second.push_back(blend);
	}
}

void NeuralNet::Mutate(SerializedNeuralNet& net, float mutationRate, float mutationAmount)
{
	for (size_t i = 0; i < net.second.size(); ++i)
	{
		// should this section mutate
		if (RandomScalar() < mutationRate)
		{
			// mutate
			float mutatedWeight = net.second[i] + ((mutationAmount * 2) * RandomScalar()) - mutationAmount;

			// clamp between 0 and 1
			if (mutatedWeight < 0)
			{
				mutatedWeight = 0;
			}
			else if (mutatedWeight > 1)
			{
				mutatedWeight = 1;
			}

			// assign new mutated weight
			net.second[i] = mutatedWeight;
		}
	}
}

void NeuralNet::SaveToFile(std::string filename) const
{
	std::ofstream savefile(filename);
	assert(savefile.is_open());

	SerializedNeuralNet vec;
	SerializeToVector(vec);

	savefile << vec.first.size() << std::endl;
	for (size_t i = 0; i < vec.first.size(); ++i)
	{
		savefile << vec.first[i] << " ";
	}
	savefile << std::endl;
	savefile << vec.second.size() << std::endl;
	for (size_t i = 0; i < vec.second.size(); ++i)
	{
		savefile << vec.second[i] << " ";
	}
	savefile << std::endl;
}



void NeuralNet::SetupFromVectors(const std::vector<size_t>& topology, const std::vector<float>& weights)
{
	// the number of layers
	size_t numLayers = topology.size();
	// index of weights for neurons to read
	size_t index = 0;

	// add each layer to the neural net
	for (size_t i = 0; i < numLayers; ++i)
	{
		// get the number of neurons in the next layer, if there is a next layer, otherwise 0
		size_t numOutputs = (i == (numLayers - 1)) ? 0 : topology[i + 1] - 1;

		// add a new layer
		m_layers.push_back(Layer());

		// add the neurons to that layer
		for (size_t j = 0; j < topology[i]; j++)
		{
			m_layers.back().push_back(Neuron(numOutputs, j));
			m_layers.back().back().ReadInWeights(weights, index);
		}

		// set the bias neuron's output to 1
		m_layers.back().back().SetOutputValue(1.0);
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

float NeuralNet::
RandomScalar()
{
	return rand() / static_cast<float>(RAND_MAX);
}
