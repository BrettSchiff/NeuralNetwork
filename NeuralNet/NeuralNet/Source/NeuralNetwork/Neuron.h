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

// first double is the weight, second is the change(delta) of weight
typedef std::pair<double, double> ConnectionWeights;

// Neurons that will make up the neural network
class Neuron
{
public:
	// constructors
	Neuron(size_t neuronsInNextLayer, size_t indexIntoCurrentLayer);

	// methods
	double GetOutputValue() const;
	void SetOutputValue(double value);

	void FeedForward(const Layer& previousLayer);

	void CalculateOutputGradients(double targetValue);
	void CalculateHiddenGradients(const Layer& nextLayer);
	void UpdateInputWeights(Layer& previousLayer);

private:
	// private methods
	static double RandomWeight();
	static double SquashFunction(double input);
	static double SquashFunctionDerivative(double input);
	double SumDOW(const Layer& nextLayer) const;

	static double eta;   // overall net training rate 0-1
	static double alpha; // momentum(incorporation of last weight change) 0-1

	// data
	size_t m_indexIntoLayer;
	double m_outputValue;
	double m_gradient;
	std::vector<ConnectionWeights> m_outputWeights;
};
