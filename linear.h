#pragma once
#include <CL/sycl.hpp>
#include "operations.h"
#include "data.h"
using namespace cl::sycl;

template<typename T, const int GROUP_SIZE>
class Linear {
public:
	T* weight;
	T* input;
	T* result;
	T* bias;
	T* dz;
	T* dw;
	const int M;
	const int N;
	const int K;

	Linear(T* x, T* r, int m, int n, int k, queue& Q) : input(x), result(r), M(m), N(n), K(k) {
		weight = malloc_device<T>(M * N, Q);
		bias = malloc_device<T>(M, Q);
		dz = malloc_device<T>(N * K, Q);
		dw = malloc_device<T>(M * N, Q);

		Q.memset(dz, 0, N * K * sizeof(T)).wait();
		Q.memset(dw, 0, M * N * sizeof(int)).wait();

		if (dz == nullptr) {
			std::cout << "allocate failed: " << dz << std::endl;
		}
	}

	void operator()(group<2> group) const {

		T tile[GROUP_SIZE];

		for (int k = 0; k < N; k += GROUP_SIZE) {

			group.parallel_for_work_item([&](h_item<2> item) {
				int m = item.get_global_id(0);
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

			result[m * K + n] = (result[m * K + n] > 0) ? result[m * K + n] : 0;
		});
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

	void update(T* diff, T alpha, queue& Q) {
		T scale = alpha / K;
		std::cout << "dz : " << dz << ", dw: " << dw << ", diff: " << diff << std::endl;

	
		auto e1 = Q.submit([&](handler& cgh) {
			// std::cout << "update bias..." << std::endl;
			Substract<T, false> sub(bias, diff, scale, M, 1);
			cgh.parallel_for_work_group(range<2>{M, 1}, { 1, 1 }, sub);
		});

		auto e2 = Q.submit([&](handler& cgh) {
			// std::cout << "compute: dz..." << std::endl;
			TMultiply<T, GROUP_SIZE> mat(weight, diff, dz, M, N, K);
			cgh.parallel_for_work_group(range<2>{ N, K / GROUP_SIZE }, { 1, GROUP_SIZE }, mat);
		});


		auto e3 = Q.submit([&](handler& cgh) {
			// std::cout << "compute dw..." << std::endl;
			// std::cout << input << std::endl;
			MultiplyT<T, GROUP_SIZE> mat(diff, input, dw, M, N, K);
			cgh.parallel_for_work_group(range<2>{ M, N / GROUP_SIZE }, { 1, GROUP_SIZE }, mat);
		});

		auto e4 = Q.submit([&](handler& cgh) {
			// std::cout << "update weight..." << std::endl;
			// must finish compute dz before updating weight
			cgh.depends_on({ e2, e3 });
			Substract<T, true> sub(weight, dw, scale, M, N);
			cgh.parallel_for_work_group(range<2>{M, N / GROUP_SIZE}, { 1, GROUP_SIZE }, sub);
		});

		e1.wait();
		e4.wait();
		std::cout << "---------dz----------" << std::endl;
		Q.submit([&](handler& cgh) {
			T* temp = dz;
			stream out(1024, 256, cgh);
			cgh.parallel_for(range<1>{N * K}, [=](id<1> i) {
				out << i << ": " << temp[i] << " ";
			});
		}).wait();
		std::cout << std::endl;
		std::cout << "--------dw----------" << std::endl;
		Q.submit([&](handler& cgh) {
			T* temp = dw;
			stream out(1024, 256, cgh);
			cgh.parallel_for(range<1>{M * N}, [=](id<1> i) {
				out << i << ": " << temp[i] << " ";
			});
		}).wait();
		std::cout << std::endl;
		std::cout << "----------diff---------" << std::endl;

		Q.submit([&](handler& cgh) {
			T* temp = diff;
			stream out(1024, 256, cgh);
			cgh.parallel_for(range<1>{M * K}, [=](id<1> i) {
				out << i << ": " << temp[i] << " ";
			});
		}).wait();
		std::cout << std::endl;
		std::cout << "--------------------" << std::endl;

		free(diff, Q);
		diff = malloc_device<T>(N * K, Q);
		if (diff == nullptr) std::cout << "null pointer" << std::endl;

		Q.memcpy(diff, dz, N * K * sizeof(T)).wait();
		Q.memset(input, 0, N * K * sizeof(T)).wait();
		Q.memset(dw, 0, M * N * sizeof(T)).wait();
		Q.memset(dz, 0, N * K * sizeof(T)).wait();
	}

};