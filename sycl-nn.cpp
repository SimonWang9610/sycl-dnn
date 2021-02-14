#include <CL/sycl.hpp>
#include <iostream>
#include <random>
#include <vector>
#include "linear.h"
#include "operations.h"

using namespace cl::sycl;

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
void print(T* v, int row, int col) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			std::cout << v[i * col + j] << " ";
		}
		std::cout << std::endl;
	}
}


int main() {
	constexpr int SIZE = 2;
	constexpr int BATCH = 2;
	queue Q{ cpu_selector{} };

	std::vector<int> config = { 10, 8, 5, 4 };
	float* output = malloc_shared<float>(4 * BATCH, Q);
	float* target_device = malloc_device<float>(4 * BATCH, Q);

	auto input = createInput<float>(10, 2);
	auto target_host = createInput<float>(4, 2);

	auto parameters = init<float>(config);
	parameters.back().print(4, 5);

	Layer<float, SIZE, BATCH> net(config, output, Q);
	net.copyToDevice(parameters, target_device, target_host, 4 * BATCH, Q);
	
	for (int i = 0; i < 4; i++) {
		net.forward(input, Q);
		print<float>(output, 4, BATCH);
		net.difference(target_device, Q);
		net.backward(1, Q);
		net.resetInputs(Q);
	}
	
	// std::cout << "computing difference..." << std::endl;
	// net.difference(target_device, Q);
	
	// std::cout << "backwarding..." << std::endl;
	// net.backward(1.0, Q);

	//std::cout << "forwarding..." << std::endl;
	//net.forward(input, Q);

	//for (int i = 0; i < BATCH * 10; i++) {
	//	std::cout << output[i] << " ";

	//	if (i == 10) {
	//		std::cout << std::endl;
	//	}
	//}

	//float* o = malloc_shared<float>(10 * BATCH, Q);
	//float* result = malloc_shared<float>(10 * BATCH, Q);

	//for (int i = 0; i < 10 * BATCH; i++) {
	//	o[i] = i % 2;
	//}

	//Q.submit([&](handler& cgh) {
	//	stream out(1024, 256, cgh);

	//	cgh.parallel_for_work_group(range<2>{10 , BATCH}, { 1, SIZE }, [=](group<2> group) {
	//		float sums[BATCH] = { 0 };

	//		out << "computing..." << endl;

	//		group.parallel_for_work_item([&](h_item<2> item) {
	//			int m = item.get_global_id(0);
	//			int n = item.get_global_id(1);

	//			if (sums[n] == 0) {
	//				for (int k = 0; k < 10; k++) {
	//					sums[n] += exp(o[k * BATCH + n]);
	//				}
	//			}

	//			out << "[" << m << ", " << n << "]: " << sums[m] << endl;
	//		});

	//		group.parallel_for_work_item([&](h_item<2> item) {
	//			int m = item.get_global_id(0);
	//			int n = item.get_global_id(1);

	//			result[m * BATCH + n] = exp(o[m * BATCH + n]) / sums[n];
	//		});
	//	});
	//});

	//Q.wait();

	//for (int i = 0; i < 10 * BATCH; i++) {
	//	std::cout << result[i] << " ";
	//}

	return 0;
}