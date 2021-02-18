#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include "layers.h"

using namespace cl::sycl;


int main() {
	constexpr int SIZE = 1;
	constexpr int batch = 2;
	constexpr float alpha = 0.01;
	queue Q{gpu_selector{}};

	std::vector<int> config = { 6, 5, 4, 3 };
	float* output = malloc_shared<float>(config.back() * batch, Q);
	float* target_device = malloc_device<float>(config.back() * batch, Q);

	auto input = createInput<float>(config[0], batch);
	auto target_host = createInput<float>(config.back(), batch);
	auto parameters = init<float>(config);

	Layer<float, SIZE, batch> net(config, output, Q);
	net.copyToDevice(parameters, target_device, target_host, config.back() * batch, Q);
	net.print();

		try {
			net.forward(input, Q);
			printOutput<float>(output, config.back(), batch);
			net.difference(target_device, Q);
			net.backward(alpha, Q);
			net.reset(Q);
		}
		catch (exception const& e) {
			std::cout << e.what() << std::endl;
			return 2;
		}

	return 0;
}
