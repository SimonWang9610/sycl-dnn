#pragma once
#include <CL/sycl.hpp>
#include <vector>
#include "operations.h"
#include "data.h"

using namespace cl::sycl;

typedef accessor<int, 2, access::mode::read, access::target::global_buffer> DataRead2;
typedef accessor<int, 2, access::mode::read_write, access::target::global_buffer> DataWrite2;
typedef accessor<int, 1, access::mode::read_write, access::target::local> DataLocal;
typedef accessor<int, 2, access::mode::read_write, access::target::local> DataLocal2;

template<typename T, const int GROUP_SIZE>
class LinearRange {
private:
	DataRead2 weight; //[a, b]
	DataRead2 input; // [b, c]
	DataRead2 bias; // [a, 1]
	DataWrite2 result; // [a, c]
	DataLocal tile;
public:
	LinearRange(DataRead2& w, DataRead2& x, DataRead2& b, DataWrite2& r, handler& cgh) :
		weight(w), input(x), bias(b), result(r), tile(DataLocal{ SIZE, cgh }) {}

	void operator()(nd_item<2> item) const {
		int m = item.get_global_id(0);
		int n = item.get_global_id(1);
		int i = item.get_local_id(1);

		T sum = 0;
		int col = weight.get_range()[1];
		int row = input.get_range()[0];

		for (int k = 0; k < col; k += GROUP_SIZE) {
			tile[i] = (k + i < col)? weight[m][k + i]: 0;
			item.barrier();

			for (int kk = 0; kk < GROUP_SIZE; kk++) {
				sum += (k + kk < row)? tile[kk] * input[k + kk][n] : 0;
				item.barrier();
			}
		}
		result[m][n] = sum + bias[m][0];
	}
};

template<typename T, const int GROUP_SIZE>
class LinearGroup {
private:
	DataRead2 weight; //[a, b]
	DataRead2 input; // [b, c]
	DataRead2 bias; // [a, 1]
	DataWrite2 result; // [a, c]

public:
	LinearGroup(DataRead2& w, DataRead2& x, DataRead2& b, DataWrite2& r) :
		weight(w), input(x), bias(b), result(r) {}

	void operator()(group<2> group) const {
		T tile[GROUP_SIZE];

		int col = weight.get_range()[1];
		int row = input.get_range()[0];
		// weight * x
		for (int k = 0; k < col; k += GROUP_SIZE) {

			group.parallel_for_work_item([&](h_item<2> item) {
				int m = item.get_global_id(0);
				int i = item.get_local_id(1);
				tile[i] = (k + i < col)? weight[m][k + i]: 0;
			});

			group.parallel_for_work_item([&](h_item<2> item) {
				int m = item.get_global_id(0);
				int n = item.get_global_id(1);

				for (int kk = 0; kk < GROUP_SIZE; kk++) {
					result[m][n] += (k + kk < row) ? tile[kk] * input[k + kk][n]: 0;
				}
			});
		}

		// weight * x + b
		group.parallel_for_work_item([&](h_item<2> item) {
			int m = item.get_global_id(0);
			int n = item.get_global_id(1);
			result[m][n] += bias[m][0];
		});
	}
};


struct Events {
	event e1;
	event e2;
};

//class Linear {
//public:
//	Linear() {}
//	virtual void operator()(group<2> group) const {}
//	virtual vector_class<event> copyToDevice(variable<float>& v, queue& Q) {}
//	virtual float* update(float* diff, float alpha, queue& Q) {}
//};

template<typename T, const int M, const int N, const int K, const int GROUP_SIZE>
class LinearUSM {
private:
	T* weight; // malloc_device
	T* input; // malloc_shared
	T* bias; // malloc_device
	T* result; // malloc_shared
	T* temp; // tile, remind to free!!!
public:
	LinearUSM(T* i, T* r, queue& Q) {
		weight = malloc_device<T>(M * N, Q);
		bias = malloc_device<T>(M, Q);
		temp = malloc_device<T>(GROUP_SIZE, Q);
		input = i;
		result = r;
	}

	vector_class<event> copyToDevice(T* w, T* b, queue& Q) {
		// cgh.depends_on(event)
		// event occassionally was not completed before executing kernel functions 
		return {
			Q.submit([&](handler& cgh) {
			cgh.memcpy(weight, w, M * N * sizeof(T));
		}),
			Q.submit([&](handler& cgh) {
			cgh.memcpy(bias, b, M * sizeof(T));
		})
		};
	}

	T* w() {
		return weight;
	}

	T* b() {
		return bias;
	}

	void operator()(nd_item<2> item) const {
		int m = item.get_global_id(0);
		int n = item.get_global_id(1);
		int i = item.get_local_id(1);

		T sum = 0;

		for (int k = 0; k < N; k += GROUP_SIZE) {
			temp[i] = (k + i < N) ? weight[m * N + k + i] : 0;
			item.barrier();

			for (int kk = 0; kk < GROUP_SIZE; kk++) {
				sum += (k + kk < N) ? temp[kk] * input[(k + kk) * K + n] : 0;
				item.barrier();
			}
		}
		/*
			occasionally, the reuslt is not consistent with plain multiplication
			it might be because the malloc_shared/malloc_device has some synchronization problem?
		*/

		//for (int k = 0; k < N; k++) {
		//	sum += weight[m * N + k] * input[k * K + n];
		//}

		result[m * K + n] = sum + bias[m];
	}

};

template<typename T, const int GROUP_SIZE>
class LinearGroupUSM {
private:
	T* weight;
	T* input;
	T* bias;
	T* result;
	const int M;
	const int N;
	const int K;
	bool end;
	//T* dz;
	//T* dw;
public:
	LinearGroupUSM(T* x, T* r, bool e, int m, int n, int k, queue& Q): input(x), result(r), 
		M(m), N(n), K(k), end(e) {
		weight = malloc_device<T>(M * N, Q);
		bias = malloc_device<T>(M, Q);
		std::cout << "input: " << input << ", output: " << result << std::endl;
	}

	vector_class<event> copyToDevice(variable<T>& v, queue& Q) const {
		return {
			Q.submit([&](handler& cgh) {
				cgh.memcpy(weight, v.weight, M * N * sizeof(T));
			}),
			Q.submit([&](handler& cgh) {
				cgh.memcpy(bias, v.bias, M * sizeof(T));
			})
		};
	}

	void operator()(group<2> group) const {

		T tile[GROUP_SIZE];

		for (int k = 0; k < N; k += GROUP_SIZE) {

			group.parallel_for_work_item([&](h_item<2> item) {
				int m = item.get_global_id(0);
				// int n = item.get_global_id(1);
				int i = item.get_local_id(1);
				tile[i] = (k + i < N) ? weight[m * N + k + i] : 0;
			});

			group.parallel_for_work_item([&](h_item<2> item) {
				int m = item.get_global_id(0);
				int n = item.get_global_id(1);

				for (int kk = 0; kk < GROUP_SIZE; kk++) {
					result[m * K + n] += (k + kk < N) ? tile[kk] * input[(k + kk) * K + n] : 0;
				}
			});
		}

		group.parallel_for_work_item([&](h_item<2> item) {
			int m = item.get_global_id(0);
			int n = item.get_global_id(1);

			result[m * K + n] += bias[m];
		});

		if (!end) {
			group.parallel_for_work_item([&](h_item<2> item) {
				int m = item.get_global_id(0);
				int n = item.get_global_id(1);

				result[m * K + n] = (result[m * K + n] > 0) ? result[m * K + n] : 0;
			});
		}
		
	}

	T* update(T* diff, T alpha, queue& Q) const {
		this->view();

		T* dz = malloc_device<T>(N * K, Q);
		T* dw = malloc_device<T>(M * N, Q);
		T scale = alpha / K;

		auto e1 = Q.submit([&](handler& cgh) {
			// std::cout << "compute: dz..." << std::endl;
			Multiply<T, GROUP_SIZE> mat(weight, diff, dz, M, N, K, true);
			cgh.parallel_for_work_group(range<2>{ N, K / GROUP_SIZE }, { 1, GROUP_SIZE }, mat);
		});

		auto e2 = Q.submit([&](handler& cgh) {
			// std::cout << "update bias..." << std::endl;
			cgh.depends_on(e1);
			Substract<T, false> sub(bias, dz, scale, M, 1);
			cgh.parallel_for_work_group(range<2>{M, 1}, { 1, 1 }, sub);
		});

		auto e3 = Q.submit([&](handler& cgh) {
			std::cout << "compute dw..." << std::endl;
			std::cout << input << std::endl;
			Multiply<T, GROUP_SIZE> mat(diff, input, dw, M, N, K, false);
			cgh.parallel_for_work_group(range<2>{ M, N / GROUP_SIZE }, { 1, GROUP_SIZE }, mat);
		});

		auto e4 = Q.submit([&](handler& cgh) {
			// std::cout << "update weight..." << std::endl;
			// must finish compute dz before updating weight
			cgh.depends_on({ e1, e3 });
			Substract<T, true> sub(weight, dw, scale, M, N);
			cgh.parallel_for_work_group(range<2>{M, N / GROUP_SIZE}, { 1, GROUP_SIZE }, sub);
		});

		e4.wait();
		e2.wait(); // waiting for updating weight and bias using [dw, dz]
		free(dw, Q);
		free(diff, Q);
		free(input, Q);

		return dz;
	}

	void reset(T* x, T* o, queue& Q) {
		input = x;
		result = o;
	}

	void view() const {
		std::cout << "Forwarded -> input: " << input << ", output: " << result << std::endl;
	}

	void print(T* w, queue& Q) const {
		Q.memcpy(w, weight, M * N * sizeof(T)).wait();

		std::cout << "*************weight after forwarding*************" << std::endl;
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				std::cout << w[i * N + j] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "*************************" << std::endl;
	}
};

template<typename T, const int SIZE, const int BATCH>
class Layer {
private:
	std::vector<LinearGroupUSM<T, SIZE>> layers;
	std::vector<T*> inputs;
	std::vector<int> neurons;
	T* output;
public:
	Layer(std::vector<int> config, T* o, queue& Q): neurons(config), output(o) {

		for (int i = 0; i < config.size(); i++) {
			inputs.push_back(malloc_device<T>(BATCH * config[i], Q));
		}

		for (int i = 1; i < config.size(); i++) {
			bool end = (i == config.size() - 1) ? true : false;

			auto linear = LinearGroupUSM<T, SIZE>(inputs[i - 1], inputs[i], 
				config[i], config[i - 1], BATCH, end, Q);
			layers.push_back(linear);
		}
	}

	void copyToDevice(std::vector<variable<T>> parameters, T* des, T* src, int n, queue& Q) {
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

	void print(int index, T* w, queue& Q) {
		layers[index].print(w, Q);
	}

	void forward(T* x, queue& Q) const {
		Q.memcpy(inputs[0], x, neurons[0] * BATCH * sizeof(T)).wait();
		std::cout << "input[0] pointer: " << inputs[0] << std::endl;

		int i;

		for (i = 1; i < neurons.size(); i++) {
			Q.submit([&](handler& cgh) {
				cgh.parallel_for_work_group(range<2>{neurons[i], BATCH / SIZE}, { 1, SIZE }, layers[i - 1]);
			}).wait();
			layers[i - 1].view();
		}
		std::cout << "@@@@forward" << std::endl;
		/*Q.submit([&](handler& cgh) {
			Softmax<T, BATCH> softmax(inputs[i - 1], neurons[i - 1]);
			cgh.parallel_for_work_group(range<2>{neurons[i-1] / SIZE, BATCH}, { SIZE, 1 }, softmax);
		}).wait();*/

		Q.memcpy(output, inputs[i-1], BATCH * neurons[i - 1] * sizeof(T)).wait();
	}

	void resetInputs(queue& Q) {

		//while (!inputs.empty()) {
		//	free(inputs.back(), Q);
		//	inputs.pop_back();
		//}

		for (int i = 0; i < neurons.size(); i++) {
			inputs.push_back(malloc_device<T>(neurons[i] * BATCH, Q));

			if (i > 0) layers[i - 1].reset(inputs[i], inputs[i - 1], Q);
		}
		std::cout << "******reallocation!" << std::endl;
	}

	void difference(T* target, queue& Q) const {
		T* out = inputs.back();

		Q.submit([&](handler& cgh) {
			cgh.parallel_for_work_group(range<2>{neurons.back(), BATCH}, { 1, SIZE }, [=](group<2> group) {
				group.parallel_for_work_item([&](h_item<2> item) {
					int m = item.get_global_id(0);
					int n = item.get_global_id(1);
					out[m * BATCH + n] -= target[m * BATCH + n];
				});
			});
		}).wait();
		// std::cout << "###Computed Difference!" << std::endl;
	}

	void backward(T alpha, queue& Q) {
		T* diff = inputs.back();
		inputs.pop_back();

		for (auto layer = layers.rbegin(); layer != layers.rend(); layer++) {
			diff = layer->update(diff, alpha, Q);
			inputs.pop_back();
			std::cout << "^^^^^Backward..." << std::endl;
		}
		free(diff, Q);
		std::cout << "COMPLETED!" << std::endl;
	}
};

