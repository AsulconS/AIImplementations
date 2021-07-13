#include <perceptron.hpp>

#include <fstream>

int main()
{
    std::vector<std::vector<float>> X {
        {0.50f, 0.20f, 0.118f},
        {0.30f, 0.15f, 0.130f},
        {0.70f, 0.10f, 0.520f},
        {0.20f, 0.30f, 0.100f}
    };
    std::vector<float> Y {1.0f, 0.0f, 0.0f, 1.0f};

    std::ofstream of;
    of.open("results.txt");
    Perceptron p {0.25f, 32, of};
    p.fit(X, Y);

    std::cout << p.predict({0.50f, 0.20f, 0.118f}) << std::endl;
    std::cout << p.predict({0.30f, 0.15f, 0.130f}) << std::endl;
    std::cout << p.predict({0.70f, 0.10f, 0.520f}) << std::endl;
    std::cout << p.predict({0.20f, 0.30f, 0.100f}) << std::endl;

    return 0;
}
