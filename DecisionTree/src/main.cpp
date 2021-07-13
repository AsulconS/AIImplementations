#include <mnistParsing.hpp>
#include <iomanip>
#include <iostream>

int main()
{
    std::vector<std::vector<uint8_t>> X;
    loadTrainingSetImageFile("train-images.idx3-ubyte", X);

    std::vector<uint8_t> Y;
    loadTrainingSetLabelFile("train-labels.idx1-ubyte", Y);

    for(int i {0}; i < 28; ++i)
    {
        for(int j {0}; j < 28; ++j)
            std::cout << std::setw(4) << (uint32_t)X[0][i * 28 + j];
        std::cout << std::endl;
    }

    return 0;
}
