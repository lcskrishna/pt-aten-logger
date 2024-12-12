cmake_minimum_required(VERSION 3.10)
project(cpp_tests)

set(CMAKE_CXX_STANDARD 14)

# find and link torch.
find_package(Torch REQUIRED)

add_executable(cpp_test example.cpp)
target_link_libraries(cpp_test "${TORCH_LIBRARIES}")

# ensure proper linking
set_property(TARGET cpp_test PROPERTY CXX_STANDARD 14)

