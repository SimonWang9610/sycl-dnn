#pragma once
#include <CL/sycl.hpp>
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

template<typename T>
std::vector<variable<T>> init(std::vector<int> config) {
	std::vector<variable<T>> params;

	for (int i = 1; i < config.size(); i++) {
		params.push_back(variable<T>(config[i], config[i - 1]));
	}

	return params;
}