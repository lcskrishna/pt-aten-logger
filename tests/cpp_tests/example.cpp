#include <torch/torch.h>

int main() {
    // Create two tensors
    torch::Tensor tensor1 = torch::randn({2, 3});
    torch::Tensor tensor2 = torch::randn({2, 3});

    // Add the tensors
    torch::Tensor result = tensor1 + tensor2;

    // Print the result
    std::cout << result << std::endl;

    return 0;
}
