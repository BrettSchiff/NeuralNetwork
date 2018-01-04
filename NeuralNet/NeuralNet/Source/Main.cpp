//////////////////////////////////////////////////////
// 
// File: Main.cpp
// Purpose: main(testing mostly)
// 
// Author: Brett Schiff
// Contact: brettschiff@gmail.com
// 
//////////////////////////////////////////////////////
#include "NeuralNetwork\NeuralNet.h"
#include "TestAverage\TestAverage.h"
#include <vector>
#include <iostream>

#define NUM_NUMBERS_IN_AVERAGE_TEST 3
#define NUM_SAMPLES_IN_AVERAGE_TEST 100000

int main()
{
	// vector of unsigneds representing number of nodes in each layer
	std::vector<size_t> topology;
	topology.push_back(NUM_NUMBERS_IN_AVERAGE_TEST);
	topology.push_back(NUM_NUMBERS_IN_AVERAGE_TEST);
	topology.push_back(NUM_NUMBERS_IN_AVERAGE_TEST);
	topology.push_back(1);

	// create a network
	NeuralNet myNet(topology);

	// data: create the random numbers
	AverageData data(NUM_NUMBERS_IN_AVERAGE_TEST, NUM_SAMPLES_IN_AVERAGE_TEST);

	for (size_t i = 0; i < NUM_SAMPLES_IN_AVERAGE_TEST; ++i)
	{
		std::vector<double> resultValues;

		// feed, train, and get results from the network
		myNet.FeedForward(data.m_data[i]);
		myNet.GetResults(resultValues);
		myNet.BackPropogation(data.m_correctAnswers[i]);

		std::cout << "Test Number " << i << std::endl;
		std::cout << "average of " << NUM_NUMBERS_IN_AVERAGE_TEST << " numbers." << std::endl;
		std::cout << "   Correct Answer: " << data.m_correctAnswers[i][0] << std::endl;
		std::cout << "Neural Net Answer: " << resultValues[0] << std::endl << std::endl;

	}

	return 0;
}