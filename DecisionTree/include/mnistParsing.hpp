#ifndef MNIST_PARSING_HPP
#define MNIST_PARSING_HPP

#include <vector>
#include <cstdint>
#include <fstream>
#include <iostream>

bool loadTrainingSetLabelFile(const char* path, std::vector<uint8_t>& Y);
bool loadTrainingSetImageFile(const char* path, std::vector<std::vector<uint8_t>>& X);

#endif // MNIST_PARSING_HPP
