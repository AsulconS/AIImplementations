#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <cmath>
#include <vector>
#include <string>
#include <iostream>

class Perceptron
{
public:
    Perceptron(float t_eta, int t_epochs, std::ostream& t_os = std::cout);
    virtual ~Perceptron();

    float predict(const std::vector<float>& X);
    void fit(const std::vector<std::vector<float>>& M_X, const std::vector<float>& Y);

private:
    float activate(float x);
    float netInput(const std::vector<float>& X);
    void printWeights(const std::string& title = "Current Weights:");

private:
    float m_eta;
    int m_epochs;
    std::vector<float> m_weights;

    std::ostream& m_os;
};

#endif // NEURAL_NETWORK_HPP
