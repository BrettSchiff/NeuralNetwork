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
#include "TestGame\TestGame.h"
#include "Windows.h"
#include <vector>
#include <iostream>

#define NUM_NUMBERS_IN_AVERAGE_TEST 3
#define NUM_SAMPLES_IN_AVERAGE_TEST 1000

#define GAME_WIDTH 120
#define GAME_HEIGHT 5
#define GAME_DIFFICULTY 2
#define GAME_TIME_BETWEEN_MOVES .1
#define NUM_GAME_TRAINING_FRAMES 2000
#define NUM_ROWS_SEEN 3
#define NUM_ROUNDS_FOR_NET_TO_PLAY 1000

int main()
{
	// Test Average
	//// vector of unsigneds representing number of nodes in each layer
	//std::vector<size_t> topology;
	//topology.push_back(NUM_NUMBERS_IN_AVERAGE_TEST);
	//topology.push_back(NUM_NUMBERS_IN_AVERAGE_TEST);
	//topology.push_back(NUM_NUMBERS_IN_AVERAGE_TEST);
	//topology.push_back(1);

	//// create a network
	//NeuralNet myNet(topology);

	//// data: create the random numbers
	//AverageData data(NUM_NUMBERS_IN_AVERAGE_TEST, NUM_SAMPLES_IN_AVERAGE_TEST);

	//for (size_t i = 0; i < NUM_SAMPLES_IN_AVERAGE_TEST; ++i)
	//{
	//	std::vector<double> resultValues;

	//	// feed, train, and get results from the network
	//	myNet.FeedForward(data.m_data[i]);
	//	myNet.GetResults(resultValues);
	//	myNet.BackPropogation(data.m_correctAnswers[i]);

	//	std::cout << "Test Number " << i << std::endl;
	//	std::cout << "average of " << NUM_NUMBERS_IN_AVERAGE_TEST << " numbers." << std::endl;
	//	std::cout << "   Correct Answer: " << data.m_correctAnswers[i][0] << std::endl;
	//	std::cout << "Neural Net Answer: " << resultValues[0] << std::endl << std::endl;

	//}

	//getchar();

	// Test Game


	std::vector<size_t> gameTopology;
	gameTopology.push_back(NUM_ROWS_SEEN * GAME_HEIGHT);
	gameTopology.push_back(GAME_HEIGHT);
	gameTopology.push_back(3);
	gameTopology.push_back(1);

	NeuralNet gameNet(gameTopology);

	Game playerGame(GAME_WIDTH, GAME_HEIGHT, GAME_DIFFICULTY);

	for (size_t i = 0; i < 1000; i++)
	{
		char input = getchar();
		int gameInput = 0;

		if (input == 'w')
		{
			gameInput = 1;
		}
		if (input == 's')
		{
			gameInput = -1;
		}
		if (input == 'e')
		{
			while (input != '\n')
			{
				input = getchar();
			}
			break;
		}

		while (input != '\n')
		{
			input = getchar();
		}

		playerGame.Update(gameInput);

		std::vector<bool> obs;

		playerGame.GetObstacleList(obs, NUM_ROWS_SEEN);

		for (size_t i = 0; i < NUM_ROWS_SEEN * GAME_HEIGHT; i++)
		{
			if (i % GAME_HEIGHT == 0)
			{
				std::cout << std::endl;
			}

			std::cout << obs[i];
		}
		std::cout << std::endl;
	}

	system("cls");
	std::cout << "\n\n\n\t\tThe Neural net is now training itself. This may take a bit\n";

	for (size_t i = 0; i < NUM_GAME_TRAINING_FRAMES; i++)
	{
		std::vector<bool> obstacles;
		double correctAnswer = .5f;
		
		// create a game
		Game sampleGame(GAME_WIDTH, GAME_HEIGHT, GAME_DIFFICULTY, 0);

		// move up to the position where there are obstacles
		for (size_t i = 0; i < EMPTY_START - 1; i++)
		{
			sampleGame.Update(0, 0);
		}

		// get the obstacles in view
		sampleGame.GetObstacleList(obstacles, NUM_ROWS_SEEN);

		// convert that to a vector of doubles
		std::vector<double> obstaclesDouble;
		size_t obstacleListSize = obstacles.size();

		for (size_t i = 0; i < obstacleListSize; i++)
		{
			if (obstacles[i])
			{
				obstaclesDouble.push_back(1.0);
			}
			else
			{
				obstaclesDouble.push_back(0.0);
			}
		}

		size_t playerPosition = GAME_HEIGHT / 2;

		if (obstacles[playerPosition] != 0)
		{
			// if the next space above the player is open, go up
			if (obstacles[playerPosition + 1] == 0)
			{
				correctAnswer = 1;
			}
			else if (obstacles[playerPosition - 1] == 0) // else if the bottom position is open, take it
			{
				correctAnswer = 0;
			}
		}

		std::vector<double> correctAnswerVector;
		correctAnswerVector.push_back(correctAnswer);

		gameNet.FeedForward(obstaclesDouble);
		gameNet.BackPropogation(correctAnswerVector);

	}

	std::cout << "\n\n\t\tAll done! Press Enter to Continue... ";
	getchar();

	// The actual game played by the Neural Net
	Game netGame(GAME_WIDTH, GAME_HEIGHT, GAME_DIFFICULTY);

	for (size_t i = 0; i < NUM_ROUNDS_FOR_NET_TO_PLAY; i++)
	{
		// get the obstacles in view
		std::vector<bool> obstacles;
		netGame.GetObstacleList(obstacles, NUM_ROWS_SEEN);

		// convert that to a vector of doubles
		std::vector<double> obstaclesDouble;
		size_t obstacleListSize = obstacles.size();

		for (size_t j = 0; j < obstacleListSize; j++)
		{
			if (obstacles[j])
			{
				obstaclesDouble.push_back(1.0);
			}
			else
			{
				obstaclesDouble.push_back(0.0);
			}
		}

		gameNet.FeedForward(obstaclesDouble);

		// get the result
		std::vector<double> netResult;
		gameNet.GetResults(netResult);

		double answer = netResult[0] - .5;

		if (answer >= -.1 && answer <= .1)
		{
			answer = 0;
		}

		netGame.Update(answer);

		Sleep(static_cast<int>(GAME_TIME_BETWEEN_MOVES * 1000));
	}

	return 0;
}