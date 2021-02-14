#pragma once
#include <CL/sycl.hpp>
#include <math.h>

using namespace cl::sycl;
template<typename T, const int SIZE>
class Multiply {
private:
	T* left;
	T* right;
	T* result;
	const int M;
	const int N;
	const int K;
	bool transpose;
public:
	Multiply(T* l, T* r, T* s, int m, int n, int k, bool t) : left(l), right(r), result(s),
	M(m), N(n), K(k), transpose(t) {}

	void operator()(group<2> group) const {
		T tile[SIZE];

		int MID_LEFT = transpose ? M: N;
		int MID_RIGHT = transpose ? N: K;

		for (int k = 0; k < MID_LEFT; k += SIZE) {
			group.parallel_for_work_item([&](h_item<2> item) {
				int m = item.get_global_id(0);
				int i = item.get_local_id(1);
				int index = transpose ? (k + i) * N + m : m * N + k + i;
				tile[i] = (k + i < MID_LEFT) ? left[index] : 0;
			});

			group.parallel_for_work_item([&](h_item<2> item) {
				int m = item.get_global_id(0);
				int n = item.get_global_id(1);

				for (int kk = 0; kk < SIZE; kk++) {
					int position = transpose ? (k + kk) * K + n : n * K + k + kk;
					// out << "right [" << position << "]: " << right[position] << endl;

					result[m * K + n] += (k + kk < MID_RIGHT) ? tile[kk] * right[position] : 0;
				}
			});
		}
	}
};

template<typename T, bool align>
class Substract {
private:
	T* left;
	T* right;
	T scale;
	const int M;
	const int N;
public:
	Substract(T* l, T* r, T a, int m, int n) : left(l), right(r), scale(a), M(m), N(n) {}

	void operator()(group<2> group) const {
		group.parallel_for_work_item([&](h_item<2> item) {
			int m = item.get_global_id(0);
			int n = item.get_global_id(1);

			if (align) {
				left[m * N + n] -= scale * right[m * N + n];
			}
			else {
				for (int kk = 0; kk < N; kk++) {
					left[m * N + n] -= scale * right[m * N + kk + n];
				}
			}
		});
	}
};

//if add at axis(0):
//	1> group_size {SIZE, 1}
//	2> sums[N]
//	3> sums[n] += exp(matrix[k * N + n])
//	4> matrix[m * N + n] = exp(matrix[m * N + n]) / sums[n]

// if add at axis(1)
//	1> group_size {1, SIZE}
//	2> sums[M]
//	3> sums[m] += exp(matrix[n * N + k])
//	4> matrix[m * N + n] = exp(matrix[m * N + n]) / sums[m]

template<typename T, const int N>
class Softmax {
private:
	T* matrix;
	const int M;
public:
	Softmax(T* mat, int m) : matrix(mat), M(m) {}
	void operator()(group<2> group) const {
		T sums[N] = { 0 };

		group.parallel_for_work_item([&](h_item<2> item) {
			int m = item.get_global_id(0);
			int n = item.get_global_id(1);

			if (sums[n] == 0) {
				for (int k = 0; k < M; k++) {
					sums[n] += exp(matrix[k * N + n]);
				}
			}
		});

		group.parallel_for_work_item([&](h_item<2> item) {
			int m = item.get_global_id(0);
			int n = item.get_global_id(1);

			matrix[m * N + n] = exp(matrix[m * N + n]) / sums[n];
		});
	}
};