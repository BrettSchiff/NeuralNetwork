//////////////////////////////////////////////////////
// 
// File: Neuron.h
// Purpose: Contains the Neuron Class
// 
// Author: Brett Schiff
// Contact: brettschiff@gmail.com
// 
//////////////////////////////////////////////////////
#pragma once
#include <vector>

class Neuron;                       // Forward declare class Neuron
typedef std::vector<Neuron> Layer;  // typedef of a layer of neurons

// first float is the weight, second is the change(delta) of weight
typedef std::pair<float, float> ConnectionWeights;

// Neurons that will make up the neural network
class Neuron
{
public:
	// constructors
	Neuron(size_t neuronsInNextLayer, size_t indexIntoCurrentLayer);

	// methods
	float GetOutputValue() const;
	void SetOutputValue(float value);

	void FeedForward(const Layer& previousLayer);

	void CalculateOutputGradients(float targetValue);
	void CalculateHiddenGradients(const Layer& nextLayer);
	void UpdateInputWeights(Layer& previousLayer);

	// serialization/"genetic" combination
	void StoreWeights_Concat(std::vector<float>& weights) const;
	void ReadInWeights(const std::vector<float>& weights, size_t& index);

private:
	// private methods
	static float RandomWeight();
	static float SquashFunction(float input);
	static float SquashFunctionDerivative(float input);
	float SumDOW(const Layer& nextLayer) const;

	static float eta;   // overall net training rate 0-1
	static float alpha; // momentum(incorporation of last weight change) 0-1

	// data
	size_t m_indexIntoLayer;
	float m_outputValue;
	float m_gradient;
	std::vector<ConnectionWeights> m_outputWeights;
};
