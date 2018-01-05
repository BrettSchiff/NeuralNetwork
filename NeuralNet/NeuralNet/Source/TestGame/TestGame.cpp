//////////////////////////////////////////////////////
// 
// File: TestGame.h
// Purpose: Contains the a test for the Neural Network with the end goal of playing a simple game
// 
// Author: Brett Schiff
// Contact: brettschiff@gmail.com
// 
//////////////////////////////////////////////////////
#include "TestGame.h"
#include <iostream>

Game::Game(size_t width, size_t height, size_t difficulty, bool drawGame) : m_width(width), m_height(height), m_difficulty(difficulty), m_playerColumn(0), m_playerHeight(height / 2)
{
	m_gameBoard.clear();

	// fill in each column
	for (size_t i = 0; i < MAX_SIZE; ++i)
	{
		// leave an empty space at the start for the player to get used to surroundings
		if (i < EMPTY_START)
		{
			PopulateEmptyColumn();
		}
		else // push a column with obstacles
		{
			PopulateObstacleColumn();
		}
	}

	if (drawGame)
	{
		DrawGameBoard();
	}
}

// methods
// draw the gameboard
void Game::DrawGameBoard()
{
	// clear the screen
	system("cls");

	// two blank lines at the top
	std::cout << std::endl << std::endl;

	// border the top of the gameboard
	for (size_t i = 0; i < m_width; i++)
	{
		std::cout << "-";
	}
	std::cout << std::endl;

	size_t drawStart = m_playerColumn;

	if (m_playerColumn < EMPTY_START)
	{
		drawStart = 0;
	}
	else
	{
		drawStart -= EMPTY_START;
	}

	// draw the game board
	for (size_t i = 0; i < m_height; i++)
	{
		for (size_t j = drawStart; j < drawStart + m_width; j++)
		{
			size_t realColumn = j;

			// account for looping in the drawing
			while (realColumn >= MAX_SIZE)
			{
				realColumn -= REPEAT_LENGTH;
			}

			// print the obstacle or free space
			if (m_gameBoard[realColumn][i])
			{
				std::cout << "#";
			}
			// draw the player if it's there
			else if (j == m_playerColumn && i == m_playerHeight)
			{
				std::cout << ">";
			}
			else
			{
				std::cout << " ";
			}
		}

		std::cout << std::endl;
	}

	// border the bottom of the gameboard
	for (size_t i = 0; i < m_width; i++)
	{
		std::cout << "-";
	}
	std::cout << std::endl << std::endl << std::endl;
}

// process the next frame, moving the player in the passed direction -1: down, 0: don't move, 1: up
void Game::Update(double direction, bool drawGame)
{
	// move one column
	m_playerColumn += 1;

	// loop back to the beginning if necessary
	if (m_playerColumn >= MAX_SIZE)
	{
		m_playerColumn = EMPTY_START;
	}

	// move in the desired direction
	if (direction)
	{
		int movement = static_cast<int>(direction / abs(direction));

		// if the player went above or below the board
		if (movement > 0 && m_playerHeight <= 0)
		{
			m_playerHeight = m_height - 1;
		}
		else if (movement < 0 && m_playerHeight >= m_height - 1)
		{
			m_playerHeight = 0;
		}
		else
		{
			m_playerHeight -= movement;
		}
	}

	if (drawGame)
	{
		// check if the player hits an obstacle
		if (m_gameBoard[m_playerColumn][m_playerHeight])
		{
			m_playerColumn = 0;
			m_playerHeight = m_height / 2;

			DrawGameBoard();
			std::cout << "You lost! Keep playing" << std::endl;
		}
		else
		{
			DrawGameBoard();
		}
	}
}


void Game::GetObstacleList(std::vector<bool>& list, size_t numberOfColumnsToGet)
{
	// clear the list
	list.clear();

	// push the scene to the list
	for (size_t i = m_playerColumn; i < m_playerColumn + numberOfColumnsToGet; i++)
	{
		// allign the row so that the player is in the middle
		WrappingInt j(0, static_cast<int>(m_height - 1), static_cast<int>(m_playerHeight - (m_height / 2)));

		WrappingInt endpoint(0, static_cast<int>(m_height - 1), static_cast<int>(j + m_height));

		do
		{
			WrappingInt column(0, MAX_SIZE - 1, static_cast<int>(i + 1));

			list.push_back(m_gameBoard[column][j]);

			j++;
		} while (j != endpoint);
	}
}

// private methods
void Game::PopulateObstacleColumn()
{
	m_gameBoard.push_back(GameColumn());
	GameColumn& newColumn = m_gameBoard.back();

	for (size_t i = 0; i < m_height; i++)
	{
		newColumn.push_back(0);
	}


	// add obstacles
	if (rand() % 10 <= m_difficulty)
	{
		for (size_t i = 0; i < (m_difficulty + 1) / 2; i++)
		{
			size_t position = rand() % m_height;

			newColumn[position] = true;
		}
	}
}

void Game::PopulateEmptyColumn()
{
	m_gameBoard.push_back(GameColumn());
	GameColumn& newColumn = m_gameBoard.back();

	for (size_t i = 0; i < m_height; i++)
	{
		newColumn.push_back(0);
	}
}




// WrappingInt

WrappingInt::WrappingInt(int start, int end, int startValue) : m_number(startValue), m_start(start), m_end(end)
{
	while (m_number > m_end)
	{
		m_number -= (m_end - m_start + 1);
	}

	while (m_number < m_start)
	{
		m_number += (m_end - m_start + 1);
	}
}

int WrappingInt::operator++(int rhs)
{
	if (m_number + 1 <= m_end)
	{
		m_number++;
		return m_number - 1;
	}
	else
	{
		int temp = m_number;
		m_number = m_start;
		return temp;
	}
}
int WrappingInt::operator--(int rhs)
{
	if (m_number - 1 >= m_start)
	{
		m_number--;
		return m_number + 1;
	}
	else
	{
		int temp = m_number;
		m_number = m_end;
		return temp;
	}
}
