#include <perceptron.hpp>

Perceptron::Perceptron(float t_eta, int t_epochs, std::ostream& t_os)
    : m_eta     {t_eta},
      m_epochs  {t_epochs},
      m_os      {t_os}
{
}

Perceptron::~Perceptron()
{
}

float Perceptron::predict(const std::vector<float>& X)
{
    return activate(netInput(X));
}

void Perceptron::fit(const std::vector<std::vector<float>>& M_X, const std::vector<float>& Y)
{
    m_weights.resize(M_X[0].size() + 1, 0.0f);
    printWeights("Initial Weights:");
    for(int e {0}; e < m_epochs; ++e)
    {
        for(int i {0}; i < M_X.size(); ++i)
        {
            float update {m_eta * (Y[i] - predict(M_X[i]))};
            for(int w {1}; w < m_weights.size(); ++w)
                m_weights[w] += update * M_X[i][w - 1];
            m_weights[0] = update;
        }
        printWeights("Epoch " + std::to_string(e) + " Weights:");
    }
}

float Perceptron::netInput(const std::vector<float>& X)
{
    float prob {m_weights[0]};
    for(int i {0}; i < X.size(); ++i)
        prob += m_weights[i + 1] * X[i];
    return prob;
}

float Perceptron::activate(float x)
{
    return x >= 0 ? 1.0f : 0.0f;
}

void Perceptron::printWeights(const std::string& title)
{
    m_os << title << std::endl;
    for(int i {0}; i < m_weights.size(); ++i)
        m_os << "w" << i << ": " << m_weights[i] << std::endl;
    m_os << std::endl;
}
