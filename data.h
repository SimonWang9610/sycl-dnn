#pragma once
#include <random>
#include <math.h>

template<typename T>
struct variable {
	T* weight;
	T* bias;

	variable(int row, int col) {
		weight = new T[row * col];
		bias = new T[row];

		std::default_random_engine generator;
		std::normal_distribution<T> distribution(0, 1);

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				weight[i * row + j] = distribution(generator);
			}
			bias[i] = 0;
		}
	}

	void print(int row, int col) {
		std::cout << "******init weight**********" << std::endl;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				std::cout << weight[i * col + j] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "***********************" << std::endl;
	}
};

// initialize parameters of each Linear
template<typename T>
std::vector<variable<T>> init(std::vector<int> config) {
	std::vector<variable<T>> params;

	for (int i = 1; i < config.size(); i++) {
		params.push_back(variable<T>(config[i], config[i - 1]));
	}

	return params;
}

// create random input for testing
template<typename T>
T* createInput(size_t row, size_t col) {
	std::default_random_engine generator;
	std::uniform_real_distribution<T> distribution(0, 1);

	T* v = new T[row * col];

	for (int i = 0; i < row * col; i++) {
		v[i] = distribution(generator);
	}

	return v;
}

template<typename T>
void printOutput(T* v, int row, int col) {
	std::cout << "******final output******" << std::endl;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			std::cout << v[i * col + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "************" << std::endl;
}
