//////////////////////////////////////////////////////
// 
// File: TestAverage.cpp
// Purpose: Contains the a test for the Neural Network with the end goal of taking the average of a set of numbers
// 
// Author: Brett Schiff
// Contact: brettschiff@gmail.com
// 
//////////////////////////////////////////////////////
#include "TestAverage.h"

AverageData::AverageData(size_t numNumbersPerSample, size_t numSamples) : m_numbersPerSample(numNumbersPerSample)
{
	// clear the data
	m_data.clear();
	m_correctAnswers.clear();

	// for the number of samples required
	for (size_t i = 0; i < numSamples; ++i)
	{
		// push a new data set
		m_data.push_back(AverageDataSet());

		// sum of the numbers
		float sum = 0;

		for (size_t j = 0; j < numNumbersPerSample; ++j)
		{
			// generate a random number and add it
			float randomNumber = RandomNumberBetween0and1();
			sum += randomNumber;

			// push it into the data
			m_data.back().push_back(randomNumber);
		}

		// the correct answer for the average
		float average = sum / static_cast<float>(numNumbersPerSample);

		// push it to the answers
		m_correctAnswers.push_back(AverageDataSet());
		m_correctAnswers.back().push_back(average);
	}
}

float AverageData::RandomNumberBetween0and1()
{
	return rand() / static_cast<float>(RAND_MAX);
}
