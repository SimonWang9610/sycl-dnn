#pragma once
#include <CL/sycl.hpp>
#include <math.h>

using namespace cl::sycl;

// left -= scale * right
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

template <typename T, const int N>
class Softmax
{
private:
	T* matrix;
	const int M;
public:
	Softmax(T *mat, int m) : matrix(mat), M(m) {}
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

// result = left.dot(right.t())
template<typename T, const int SIZE>
class MultiplyT {
private:
	T* left; // [M, K], read as [M, K]
	T* right; // [N, K], read as [K, N]
	T* result; // [M, N]
	const int M;
	const int N;
	const int K;
public:
	MultiplyT(T* l, T* r, T* s, int m, int n, int k) : left(l), right(r), result(s),
		M(m), N(n), K(k) {}

	void operator()(group<2> group) const {
		T tile[SIZE];

		for (int k = 0; k < K; k += SIZE) {
			group.parallel_for_work_item([&](h_item<2> item) {
				int m = item.get_global_id(0);
				int i = item.get_local_id(1);
				tile[i] = (k + i < K) ? left[(m * K)  + k + i] : 0;
			});

			group.parallel_for_work_item([&](h_item<2> item) {
				int m = item.get_global_id(0);
				int n = item.get_global_id(1);

				for (int kk = 0; kk < SIZE; kk++) {
					result[m * N + n] += (k + kk < K) ? tile[kk] * right[n * K + k + kk] : 0;
				}
				
			});
		}
	}
};

// result = left.t().dot(right) * derivate
template<typename T, const int SIZE>
class TMultiply {
private:
	T* left; // [M, N], read as [N, M]
	T* right; // [M, K], read as [M, K]
	T* result; // [N, K]
	T *derivate;
	const int M;
	const int N;
	const int K;
public:
	TMultiply(T *l, T *r, T *s, T *d, int m, int n, int k) : left(l), right(r), result(s), derivate(d),
															 M(m),
															 N(n), K(k) {}

	void operator()(group<2> group) const {
		T tile[SIZE];

		for (int k = 0; k < M; k += SIZE) {
			group.parallel_for_work_item([&](h_item<2> item) {
				int m = item.get_global_id(0);
				int i = item.get_local_id(1);
				tile[i] = (k + i < M) ? left[(k + i) * N + m] : 0;
			});

			group.parallel_for_work_item([&](h_item<2> item) {
				int m = item.get_global_id(0);
				int n = item.get_global_id(1);

				for (int kk = 0; kk < SIZE; kk++) {
					result[m * K + n] += (k + kk < M) ? tile[kk] * right[(k + kk) * K + n] : 0;
				}
			});
		}

		group.parallel_for_work_item([&](h_item<2> item) {
			int m = item.get_global_id(0);
			int n = item.get_global_id(1);
			int scale = (derivate[m * K + n] > 0) ? 1 : 0;
			result[m * K + n] *= scale;
		});
	}
};