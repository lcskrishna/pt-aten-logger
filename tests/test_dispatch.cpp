#include <iostream>
#include <utility>

template <typename T>
void printArgs(const T& arg)
{
    std::cout << arg << std::endl;
}

void anotherFunction(int a, const std::string& val)
{
    std::cout << "Entered another function:" << std::endl;
    std::cout << a << std::endl;
    std::cout << val << std::endl;
}

template<typename... Args>
void first_function(Args&&... args)
{
    // Print the args.
    std::cout << "Inside first function to print" << std::endl;
    (printArgs(std::forward<Args>(args)), ... );

    std::cout << "Forwarding to another function" << std::endl;
    anotherFunction(std::forward<Args>(args)...);
}

int main(int argc, const char* argv[])
{
    int a = 3;
    std::string val = "hello";
    first_function(a, val);
    return 0;
}
