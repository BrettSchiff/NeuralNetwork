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
#include <string>
#include "Neuron.h"

#define DEFAULT_MUTATION_RATE   .1f
#define DEFAULT_MUTATION_AMOUNT .2f

typedef std::vector<Neuron> Layer;  // typedef of a layer of neurons

class NeuralNet
{
public:
	typedef std::pair<std::vector<size_t>, std::vector<float>> SerializedNeuralNet;

	// constructors
	NeuralNet(const std::vector<size_t>& topology);
	NeuralNet(const std::vector<size_t>& topology, const std::vector<float>& weights);
	NeuralNet(std::string filename);

	// methods
	void FeedForward(const std::vector<float>& inputs);
	void BackPropogation(const std::vector<float>& targetValues);
	void GetResults(std::vector<float>& resultValues) const;

	// serialization/"genetic" combination
	void SerializeToVector(SerializedNeuralNet& serializedNet) const;
	static void GeneticBlend(SerializedNeuralNet& const parent1, SerializedNeuralNet& const parent2, SerializedNeuralNet& child);
	static void Mutate(SerializedNeuralNet& net, float mutationRate = DEFAULT_MUTATION_RATE, float mutationAmount = DEFAULT_MUTATION_AMOUNT);
	void SaveToFile(std::string filename) const;

private:
	// setup from vectors
	void SetupFromVectors(const std::vector<size_t>& topology, const std::vector<float>& weights);

	// private methods
	size_t NumberOfLayers() const;
	size_t NumberOfNeuronsInLayer(size_t layerNumber) const;
	static size_t NumberOfNeuronsInLayer(const Layer& layer);
	static float RandomScalar();

	float m_error;              // error of the current form of the net
	std::vector<Layer> m_layers; // layers of neurons
};

