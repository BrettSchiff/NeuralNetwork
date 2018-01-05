//////////////////////////////////////////////////////
// 
// File: TestGame.h
// Purpose: Contains the a test for the Neural Network with the end goal of playing a simple game
// 
// Author: Brett Schiff
// Contact: brettschiff@gmail.com
// 
//////////////////////////////////////////////////////
#pragma once
#include <vector>

#define REPEAT_LENGTH 200
#define EMPTY_START 10
#define MAX_SIZE REPEAT_LENGTH + EMPTY_START

typedef std::vector<bool> GameColumn;

class Game
{
public:
	// constructors: difficulty on a scale from 1-10
	Game(size_t width = 80, size_t height = 8, size_t difficulty = 3, bool drawGame = 1);


	// methods
	// process the next frame, moving the player in the passed direction -1: down, 0: don't move, 1: up
	void Update(double direction, bool drawGame = 1);

	void GetObstacleList(std::vector<bool>& list, size_t numberOfColumnsToGet);

private:
	// private methods
	void PopulateObstacleColumn();
	void PopulateEmptyColumn();
	void DrawGameBoard();

	// data
	size_t m_width;
	size_t m_height;
	size_t m_difficulty;

	size_t m_playerColumn;
	size_t m_playerHeight;

	std::vector<GameColumn> m_gameBoard;
};

class WrappingInt
{
public:
	WrappingInt(int start, int end, int startValue);

	int operator++(int rhs);
	int operator--(int rhs);
	operator int() { return m_number; }

private:
	int m_number;
	int m_start;
	int m_end;
};
