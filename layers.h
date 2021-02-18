#pragma once
#include <CL/sycl.hpp>
#include "linear.h"

using namespace cl::sycl;
class Difference;

template<typename T, const int SIZE, const int BATCH>
class Layer {
public:
	std::vector<Linear<T, SIZE>> layers;
	std::vector<T*> inputs;
	std::vector<int> neurons;
	T* output;

	Layer(std::vector<int> config, T* o, queue& Q) : neurons(config), output(o) {
		// allocate memory for inputs of each Linear
		for (int i = 0; i < config.size(); i++)
		{
			inputs.push_back(malloc_device<T>(BATCH * config[i], Q));
		}

		// set inputs[i-1] and intpus[i] as 'input' and 'result' of Linear
		// and 'weight' has shape [config[i], config[i-1]], 'bias' has shape [config[i]]
		// 'end' is used to determine whether the current Linear is the last one
		for (int i = 1; i < config.size(); i++) {
			int end = (i == config.size() - 1) ? 1 : 0;
			auto linear = Linear<T, SIZE>(inputs[i - 1], inputs[i],
										  config[i], config[i - 1], BATCH, end, Q);
			layers.push_back(linear);
		}
	}

	void copyToDevice(std::vector<variable<T>> parameters, T* des, T* src, int n, queue& Q) {
		// explicitly copy parameters (weight and bias) of Linear to Device
		// des is used to store the 'target_device`
		// src is used to store the 'target_host`
		vector_class<event> events;

		for (int i = 0; i < layers.size(); i++) {
			auto temp = layers[i].copyToDevice(parameters[i], Q);
			events.insert(events.end(), temp.begin(), temp.end());
		}

		auto e = Q.memcpy(des, src, n * sizeof(T));
		events.push_back(e);

		for (auto e = events.begin(); e != events.end(); e++) {
			e->wait();
		}
	}

	void forward(T* x, queue& Q) const {
		// copy input to Device
		Q.memcpy(inputs[0], x, neurons[0] * BATCH * sizeof(T)).wait();
		int i;

		for (i = 1; i < neurons.size(); i++) {
			Q.submit([&](handler& cgh) {
				cgh.parallel_for_work_group(range<2>{neurons[i], BATCH / SIZE}, { 1, SIZE }, layers[i - 1]);
			}).wait();
		}

		// Q.submit([&](handler& cgh) {
		// 	Softmax<T, BATCH> softmax(inputs[i - 1], neurons[i - 1]);
		// 	cgh.parallel_for_work_group(range<2>{neurons[i-1] / SIZE, BATCH}, { SIZE, 1 }, softmax);
		// }).wait();

		// copy inputs.back() to Host
		Q.memcpy(output, inputs[i - 1], BATCH * neurons[i - 1] * sizeof(T)).wait();
		std::cout << "complete forward!" << std::endl;
	}

	void difference(T* target, queue& Q) const {
		// compute the delta between the final output and target_device
		T* out = inputs.back();

		Q.submit([&](handler &cgh) {
			 cgh.parallel_for_work_group<class Difference>(range<2>{neurons.back(), BATCH}, {1, SIZE}, [=](group<2> group) {
				 group.parallel_for_work_item([&](h_item<2> item) {
					 int m = item.get_global_id(0);
					 int n = item.get_global_id(1);
					 out[m * BATCH + n] -= target[m * BATCH + n];
				 });
			 });
		 }).wait();
		std::cout << "complete difference!" << std::endl;
	}

	void backward(T alpha, queue& Q) {
		// update the parameters of each Linear
		// use 'diff' as temporay variable to store the delta calculated at each Linear
		// 'diff' will be change in Linear.update()
		T* diff = malloc_device<T>(neurons.back() * BATCH, Q);
		Q.memcpy(diff, inputs.back(), neurons.back() * BATCH * sizeof(T)).wait();

		for (int i = layers.size() - 1; i >= 0; i--) {
			layers[i].update(diff, alpha, Q);
		}
		free(diff, Q);
		std::cout << "COMPLETED!" << std::endl;
	}

	void reset(queue& Q) {
		// set all inputs as 0 after backwarding
		vector_class<event> events;

		for (int i = 0; i < inputs.size(); i++) {
			events.push_back(Q.memset(inputs[i], 0, neurons[i] * BATCH * sizeof(T)));
		}

		for (int i = 0; i < events.size(); i++) {
			events[i].wait();
		}
	}

	void print() {
		std::cout << "******Network configuration******" << std::endl;
		for (int i = 0; i < layers.size(); i++) {
			std::cout <<"[" << i << "]: ###" << std::endl;
			std::cout << "Weight: " << layers[i].weight \
				<< ", input: " << layers[i].input \
				<< ", output: " << layers[i].result \
				<< ", bias: " << layers[i].bias \
				<< ", dz: " << layers[i].dz \
				<< ", dw: " << layers[i].dw << std::endl;
		}
		std::cout << "************" << std::endl;
	}
};