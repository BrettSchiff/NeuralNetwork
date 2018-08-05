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
#include <algorithm>

#define NUM_NUMBERS_IN_AVERAGE_TEST 3
#define NUM_SAMPLES_IN_AVERAGE_TEST 1000

#define GAME_WIDTH 120
#define GAME_HEIGHT 6
#define GAME_DIFFICULTY 4
#define GAME_TIME_BETWEEN_MOVES .1
#define NUM_GAME_TRAINING_FRAMES 50000
#define NUM_ROWS_SEEN 3
#define NUM_ROUNDS_FOR_NET_TO_PLAY 50

// function prototypes
void TestAverage();
void PlayGame();
void RunTestGame();
void TestSerializeWithGame();

int main()
{
	// Test Average
	//TestAverage();

	// Player Play Game
	//PlayGame();

	// Test Game
	//RunTestGame();

	// Test SerializeVector
	TestSerializeWithGame();


	return 0;
}

void TestAverage()
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
		std::vector<float> resultValues;

		// feed, train, and get results from the network
		myNet.FeedForward(data.m_data[i]);
		myNet.GetResults(resultValues);
		myNet.BackPropogation(data.m_correctAnswers[i]);

		std::cout << "Test Number " << i << std::endl;
		std::cout << "average of " << NUM_NUMBERS_IN_AVERAGE_TEST << " numbers." << std::endl;
		std::cout << "   Correct Answer: " << data.m_correctAnswers[i][0] << std::endl;
		std::cout << "Neural Net Answer: " << resultValues[0] << std::endl << std::endl;

	}
}

void RunTestGame_Train(NeuralNet& gameNet)
{
	for (size_t i = 0; i < NUM_GAME_TRAINING_FRAMES; i++)
	{
		std::vector<bool> obstacles;
		int correctAnswer = 1;

		// create a game
		Game sampleGame(GAME_WIDTH, GAME_HEIGHT, GAME_DIFFICULTY, 0);

		// move up to the position where there are obstacles
		for (size_t i = 0; i < EMPTY_START - 1; i++)
		{
			sampleGame.Update(0, 0);
		}

		// get the obstacles in view
		sampleGame.GetObstacleList(obstacles, NUM_ROWS_SEEN);

		// convert that to a vector of floats
		std::vector<float> obstaclesfloat;
		size_t obstacleListSize = obstacles.size();

		for (size_t i = 0; i < obstacleListSize; i++)
		{
			if (obstacles[i])
			{
				obstaclesfloat.push_back(1.0);
			}
			else
			{
				obstaclesfloat.push_back(0.0);
			}
		}

		size_t playerPosition = GAME_HEIGHT / 2;

		if (obstacles[playerPosition] != 0)
		{
			// if the next space above the player is open, go up
			if (obstacles[playerPosition + 1] == 0)
			{
				correctAnswer = 2;
			}
			else if (obstacles[playerPosition - 1] == 0) // else if the bottom position is open, take it
			{
				correctAnswer = 0;
			}
		}

		std::vector<float> correctAnswerVector(3, 0.0);
		correctAnswerVector[correctAnswer] = 1.0;

		gameNet.FeedForward(obstaclesfloat);
		gameNet.BackPropogation(correctAnswerVector);

	}

	
}

void RunTestGame_Play(NeuralNet& gameNet)
{
	// The actual game played by the Neural Net
	Game netGame(GAME_WIDTH, GAME_HEIGHT, GAME_DIFFICULTY);

	for (size_t i = 0; i < NUM_ROUNDS_FOR_NET_TO_PLAY; i++)
	{
		// get the obstacles in view
		std::vector<bool> obstacles;
		netGame.GetObstacleList(obstacles, NUM_ROWS_SEEN);

		// convert that to a vector of floats
		std::vector<float> obstaclesfloat;
		size_t obstacleListSize = obstacles.size();

		for (size_t j = 0; j < obstacleListSize; j++)
		{
			if (obstacles[j])
			{
				obstaclesfloat.push_back(1.0);
			}
			else
			{
				obstaclesfloat.push_back(0.0);
			}
		}

		gameNet.FeedForward(obstaclesfloat);

		// get the result
		std::vector<float> netResult;
		gameNet.GetResults(netResult);

		float answer = 0;

		auto bestFit = std::max_element(netResult.begin(), netResult.end());

		answer = static_cast<float>((std::distance(netResult.begin(), bestFit) - 1) * -1);

		netGame.Update(answer);

		for (int i = 0; i < netResult.size(); ++i)
		{
			std::cout << netResult[i] << std::endl;
		}

		Sleep(static_cast<int>(GAME_TIME_BETWEEN_MOVES * 1000));
	}
}

void RunTestGame(NeuralNet& gameNet)
{
	system("cls");
	std::cout << "\n\n\n\t\tThe Neural net is now training itself. This may take a bit\n";
	RunTestGame_Train(gameNet);
	std::cout << "\n\n\t\tAll done! Press Enter to Continue... ";
	getchar();

	RunTestGame_Play(gameNet);
}

void RunTestGame()
{
	std::vector<size_t> gameTopology;
	gameTopology.push_back(NUM_ROWS_SEEN * GAME_HEIGHT);
	gameTopology.push_back(GAME_HEIGHT);
	gameTopology.push_back(3);

	NeuralNet gameNet(gameTopology);

	RunTestGame(gameNet);
}

void PlayGame()
{
	Game playerGame(GAME_WIDTH, GAME_HEIGHT, GAME_DIFFICULTY);

	for (size_t i = 0; i < 1000; i++)
	{
		char input = getchar();
		float gameInput = 0;

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
}

void TestSerializeWithGame()
{
	// create the initial neural network
	std::vector<size_t> gameTopology;
	gameTopology.push_back(NUM_ROWS_SEEN * GAME_HEIGHT);
	gameTopology.push_back(GAME_HEIGHT);
	gameTopology.push_back(3);
	NeuralNet gameNet(gameTopology);

	// train the network
	RunTestGame_Train(gameNet);
	RunTestGame_Play(gameNet);

	// serialize the result
	NeuralNet::SerializedNeuralNet result;
	gameNet.SerializeToVector(result);

	for (size_t i = 0; i < result.first.size(); i++)
	{
		std::cout << result.first[i] << " ";
	}
	std::cout << "\n\n";
	for (size_t i = 0; i < result.second.size(); i++)
	{
		std::cout << result.second[i] << " ";
	}
	std::cout << "\n\n\n\n";

	// build the net from the serialized result
	NeuralNet savedNet(result.first, result.second);

	RunTestGame_Play(savedNet);
	
	// serialize the result again to make sure it's the same
	NeuralNet::SerializedNeuralNet result2;
	savedNet.SerializeToVector(result2);

	for (size_t i = 0; i < result2.first.size(); i++)
	{
		std::cout << result2.first[i] << " ";
	}
	std::cout << "\n\n";
	for (size_t i = 0; i < result2.second.size(); i++)
	{
		std::cout << result2.second[i] << " ";
	}
	std::cout << "\n";
}
