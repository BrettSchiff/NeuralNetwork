//////////////////////////////////////////////////////
// 
// File: Neuron.cpp
// Purpose: Contains the Neuron Class
// 
// Author: Brett Schiff
// Contact: brettschiff@gmail.com
// 
//////////////////////////////////////////////////////
#include <cmath>
#include "Neuron.h"

float Neuron::eta = 0.15;  // overall net training rate 0-1
float Neuron::alpha = 0.5;	// momentum(incorporation of last weight change) 0-1

// constructors
Neuron::Neuron(size_t neuronsInNextLayer, size_t indexIntoCurrentLayer) : m_indexIntoLayer(indexIntoCurrentLayer), m_gradient(0)
{
	for (size_t i = 0; i < neuronsInNextLayer; i++)
	{
		float weight = RandomWeight();
		float weightDelta = 0;

		m_outputWeights.push_back(ConnectionWeights(weight, weightDelta));
	}
}

// methods
float Neuron::GetOutputValue() const
{
	return m_outputValue;
}

void Neuron::SetOutputValue(float value)
{
	m_outputValue = value;
}

void Neuron::FeedForward(const Layer& previousLayer)
{
	float cumulativeSum = 0;

	// automatically includes the bias because it is a neuron in the previous layer
	for (size_t i = 0; i < previousLayer.size(); i++)
	{
		const Neuron& previousNeuron = previousLayer[i];

		// multiply the previous neuron's value by its weighted connection to the current neuron
		cumulativeSum += previousNeuron.GetOutputValue() *
		                 previousNeuron.m_outputWeights[m_indexIntoLayer].first;
	}

	SetOutputValue(Neuron::SquashFunction(cumulativeSum));
}

void Neuron::CalculateOutputGradients(float targetValue)
{
	float difference = targetValue - GetOutputValue();
	m_gradient = difference * Neuron::SquashFunctionDerivative(m_outputValue);
}

void Neuron::CalculateHiddenGradients(const Layer& nextLayer)
{
	float dow = SumDOW(nextLayer);

	m_gradient = dow * Neuron::SquashFunctionDerivative(m_outputValue);
}

void Neuron::UpdateInputWeights(Layer& previousLayer)
{
	// update the weights in the previous layer
	for (size_t i = 0; i < previousLayer.size() - 1; ++i)
	{
		Neuron& currentNeuron = previousLayer[i];

		float oldDeltaWeight = currentNeuron.m_outputWeights[m_indexIntoLayer].second;

		// this can be modified in different ways
		float newDeltaWeight =
			  eta   // overall net learning rate from 0 to 1
			* currentNeuron.GetOutputValue()
			* m_gradient
			+ alpha * oldDeltaWeight;  // alpha is momentum from 0 to 1

		currentNeuron.m_outputWeights[m_indexIntoLayer].second = newDeltaWeight;
		currentNeuron.m_outputWeights[m_indexIntoLayer].first += newDeltaWeight;
	}
}


// private methods
float Neuron::RandomWeight()
{
	// return random number between - and 1
	return rand() / static_cast<float>(RAND_MAX);
}

float Neuron::SquashFunction(float input)
{
	//!?!? switch to the sigmoid function later

	// using hyperbolic tangent as squish function
	return tanh(input);
}

float Neuron::SquashFunctionDerivative(float input)
{
	//!?!? switch to the sigmoid function later

	// d/dx(tanh(x)) = 1 - tanh^2(x)
	float tanhInput = tanh(input);

	return 1 - (tanhInput * tanhInput);
}

float Neuron::SumDOW(const Layer& nextLayer) const
{
	float sum = 0;

	// sum the contributions to the errors of the nodes this layer feeds
	for (size_t i = 0; i < nextLayer.size() - 1; ++i)
	{
		sum += m_outputWeights[i].first * nextLayer[i].m_gradient;
	}

	return sum;
}
