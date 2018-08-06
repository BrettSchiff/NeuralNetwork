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
#define GAME_DIFFICULTY 3
#define GAME_TIME_BETWEEN_MOVES .05
#define NUM_GAME_TRAINING_FRAMES 50000
#define NUM_ROWS_SEEN 3
#define NUM_ROUNDS_FOR_NET_TO_PLAY 1000000

#define GEN_NUM_GENERATIONS 1000
#define GEN_NETS_PER_GEN 300
#define GEN_NUM_OF_BEST_TO_KEEP 150
#define GEN_ROUNDS_TO_TEST 500


// function prototypes
void TestAverage();
void PlayGame();
void RunTestGame();
void TestSerializeWithGame();
void TestGeneticsWithGame();
void TestLoadWithGame(std::string filename);

int main()
{
	// Test Average
	//TestAverage();

	// Player Play Game
	//PlayGame();

	// Test Game
	//RunTestGame();

	// Test SerializeVector
	//TestSerializeWithGame();

	// Test Genetic Selection
	//TestGeneticsWithGame();

	// Test loading net from file
	TestLoadWithGame("SavedNeuralNet");

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

void TestGameCreateTopology(std::vector<size_t>& gameTopology)
{
	gameTopology.push_back(NUM_ROWS_SEEN * GAME_HEIGHT);
	gameTopology.push_back(3);
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
			printf("%.3f\n", netResult[i]);
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
	TestGameCreateTopology(gameTopology);
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
	TestGameCreateTopology(gameTopology);
	NeuralNet gameNet(gameTopology);

	// train the network
	RunTestGame_Train(gameNet);
	//RunTestGame_Play(gameNet);

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

	//RunTestGame_Play(savedNet);
	
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

// int for score, neural net for neural net
typedef std::pair<float, NeuralNet> Individual;
typedef std::vector<Individual> Generation;

class IndividualSorter
{
public:
	boolean operator()(Individual& const lhs, Individual& const rhs)
	{
		return lhs.first > rhs.first;
	}
};

void TestLoadWithGame(std::string filename)
{
	NeuralNet loaded(filename);

	RunTestGame_Play(loaded);
}

//#pragma optimize("", off)
// func prototypes
void PrepareNextGen(Generation& oldGen);
void TestCurrentGen(Generation& currGen);

size_t gen;

void TestGeneticsWithGame()
{
	std::cout << "\n\n\t\tThe nets are being trained. This could take a while...\n";

	// create the initial neural net
	Generation Nets;
	for (size_t i = 0; i < GEN_NETS_PER_GEN; i++)
	{
		std::vector<size_t> gameTopology;
		TestGameCreateTopology(gameTopology);
		Nets.push_back(std::make_pair(0, NeuralNet(gameTopology)));
	}


	for (gen = 0; gen < GEN_NUM_GENERATIONS; ++gen)
	{
		TestCurrentGen(Nets);
		PrepareNextGen(Nets);
	}

	// at this point, our best neural net should be in the top slot of Nets
	Nets[0].second.SaveToFile("SavedNeuralNet");

	std::cout << "\n\n\t\tThe nets have been trained. Here is the best. Press Enter to Continue\n";
	getchar();
	RunTestGame_Play(Nets[0].second);
}

void PrepareNextGen(Generation& oldGen)
{
	std::sort(oldGen.begin(), oldGen.end(), IndividualSorter());

	// CULL THE WEAK
	while (oldGen.size() > GEN_NUM_OF_BEST_TO_KEEP)
	{
		oldGen.pop_back();
	}

	// serialize the old neural nets into breedable form
	std::vector<NeuralNet::SerializedNeuralNet> NeuralNetVecs;
	for (size_t i = 0; i < oldGen.size(); i++)
	{
		NeuralNet::SerializedNeuralNet result;
		oldGen[i].second.SerializeToVector(result);
		NeuralNetVecs.push_back(result);
	}

	// "breed" in new nets from the best, starting with the best
	//!?!? there may be a better way to select these, possibly even randomly
	int diff = 1;
	while (oldGen.size() < GEN_NETS_PER_GEN)
	{
		for (size_t i = 0; i < NeuralNetVecs.size() - diff; i++)
		{
			NeuralNet::SerializedNeuralNet newNet;
			NeuralNet::GeneticBlend(NeuralNetVecs[i], NeuralNetVecs[i + diff], newNet);
			NeuralNet::Mutate(newNet);
			Individual newInd = std::make_pair(0, NeuralNet(newNet.first, newNet.second));
			oldGen.push_back(newInd);

			if (oldGen.size() >= GEN_NETS_PER_GEN)
			{
				break;
			}
		}
		++diff;
	}
}

void TestGen_Game(Individual& gameNet, Game& netGame)
{
	// zero out the score
	gameNet.first = 0;

	for (size_t i = 0; i < GEN_ROUNDS_TO_TEST; i++)
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

		gameNet.second.FeedForward(obstaclesfloat);

		// get the result
		std::vector<float> netResult;
		gameNet.second.GetResults(netResult);

		float answer = 0;

		auto bestFit = std::max_element(netResult.begin(), netResult.end());

		answer = static_cast<float>((std::distance(netResult.begin(), bestFit) - 1) * -1);

		//bool lost = netGame.Update(answer, gen > GEN_NUM_GENERATIONS - 2);
		bool lost = netGame.Update(answer, false);

		// it costs some points to move
		if (answer != 0)
		{
			gameNet.first -= .3f;
		}

		// if lost, end the game at whatever score the net currently has
		if (lost)
		{
			return;
		}
		// else increase score
		else
		{
			++gameNet.first;
		}
	}
}

void TestCurrentGen(Generation& currGen)
{
	// The actual game played by the Neural Net
	Game netGame(GAME_WIDTH, GAME_HEIGHT, GAME_DIFFICULTY, false);

	for (size_t i = 0; i < currGen.size(); i++)
	{
		TestGen_Game(currGen[i], netGame);
	}
}
