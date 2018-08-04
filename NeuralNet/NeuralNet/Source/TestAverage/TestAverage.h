//////////////////////////////////////////////////////
// 
// File: TestAverage.h
// Purpose: Contains the a test for the Neural Network with the end goal of taking the average of a set of numbers
// 
// Author: Brett Schiff
// Contact: brettschiff@gmail.com
// 
//////////////////////////////////////////////////////
#pragma once
#include <vector>

typedef std::vector<float> AverageDataSet;

class AverageData
{
public:
	AverageData(size_t numNumbersPerSample, size_t numSamples);

	// private methods
	static float RandomNumberBetween0and1();


	// data
	size_t m_numbersPerSample;

	std::vector<AverageDataSet> m_data;
	std::vector<AverageDataSet> m_correctAnswers;
};