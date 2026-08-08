//
// Created by cartercpp on 8/7/26.
//
#include "neural_network.h"
#include <stdexcept>
#include <initializer_list>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <ranges>
#include <utility>
#include <cmath>
#include <cstddef>
#include "math_vector.h"
#include "matrix.h"

double neural_network::Relu(double x) noexcept
{
    return (x > 0) ? x : 0;
}

double neural_network::ReluDerivative(double x) noexcept
{
    return x > 0;
}

neural_network::neural_network(std::initializer_list<std::size_t> neuronsPerLayer, double learningRate)
    : m_neuronsPerLayer{neuronsPerLayer}, m_learningRate{learningRate}
{
    if (neuronsPerLayer.size() < 3)
        throw std::invalid_argument{"Neural network must have at least an input, hidden, and output layer"};

    m_weights.reserve(neuronsPerLayer.size());
    m_biases.reserve(neuronsPerLayer.size());

    std::random_device rd;

    for (std::size_t layer = 1; layer < neuronsPerLayer.size(); ++layer)
    {
        const std::size_t layerSize = *(neuronsPerLayer.begin() + layer),
                          prevLayerSize = *(neuronsPerLayer.begin() + layer - 1);

        const double stddev = std::sqrt(2.0 / static_cast<double>(prevLayerSize));
        std::normal_distribution<double> dist(0.0, stddev);

        matrix<double> weights(layerSize, math_vector<double>(prevLayerSize, 0));
        for (std::size_t i = 0; i < layerSize; ++i)
            for (std::size_t i2 = 0; i2 < prevLayerSize; ++i2)
                weights[i][i2] = dist(rd);

        m_weights.emplace_back(std::move(weights));
        m_biases.emplace_back(layerSize, 0);
    }
}

std::vector<math_vector<double>> neural_network::Forward(const math_vector<double>& input) const
{
    if (input.size() != m_neuronsPerLayer[0])
        throw std::invalid_argument{"Incorrect # of inputs given"};

    std::vector<math_vector<double>> activations;
    activations.reserve(m_neuronsPerLayer.size());
    activations.push_back(input);

    for (std::size_t layer = 0; layer < m_neuronsPerLayer.size() - 1; ++layer)
    {
        const math_vector<double>& sourceActivations{activations[layer]};
        const matrix<double>& sourceWeights{m_weights[layer]};
        const math_vector<double>& sourceBiases{m_biases[layer]};
        math_vector<double> destActivations{sourceWeights * sourceActivations + sourceBiases};

        if (layer < m_neuronsPerLayer.size() - 2)
        {
            for (std::size_t i = 0; i < destActivations.size(); ++i)
                destActivations[i] = Relu(destActivations[i]);
        }
        else
        {
            double maxActivation = std::numeric_limits<double>::lowest();

            for (std::size_t i = 0; i < destActivations.size(); ++i)
                maxActivation = std::max(maxActivation, destActivations[i]);

            destActivations -= math_vector<double>(destActivations.size(), maxActivation);

            double softmaxSum = 0;
            for (std::size_t i = 0; i < destActivations.size(); ++i)
            {
                destActivations[i] = std::exp(destActivations[i]);
                softmaxSum += destActivations[i];
            }

            for (std::size_t i = 0; i < destActivations.size(); ++i)
                destActivations[i] /= softmaxSum;
        }

        activations.emplace_back(std::move(destActivations));
    }

    return activations;
}

math_vector<double> neural_network::predict(const math_vector<double>& input) const
{
    return Forward(input).back();
}

void neural_network::fit(const math_vector<double>& input, const math_vector<double>& target)
{
    if (target.size() != m_neuronsPerLayer.back())
        throw std::invalid_argument{"Target vector size is wrong"};

    const std::vector<math_vector<double>> activations{Forward(input)};
    const math_vector<double>& prediction{activations.back()};

    math_vector<double> delta(prediction.size(), 0);
    for (std::size_t i = 0; i < prediction.size(); ++i)
        delta[i] = prediction[i] - target[i];

    for (std::size_t i = 0; i < m_weights.size(); ++i)
    {
        const std::size_t layer = m_weights.size() - i - 1;
        const math_vector<double>& layerActivations{activations[layer]};
        m_weights[layer] -= m_learningRate * outer_product(delta, layerActivations);
        m_biases[layer] -= m_learningRate * delta;

        math_vector<double> activationDerivatives(layerActivations.size(), 0);
        for (std::size_t neuron = 0; neuron < activationDerivatives.size(); ++neuron)
            activationDerivatives[neuron] = ReluDerivative(layerActivations[neuron]);

        if (layer > 0)
        {
            delta = m_weights[layer].transpose() * delta;
            delta = delta.multiply(activationDerivatives);
        }
    }
}

neural_network::neural_network(const std::vector<matrix<double>>& weights,
                                const std::vector<math_vector<double>>& biases,
                                const std::vector<std::size_t>& neuronsPerLayer,
                                double learningRate)
                                    : m_weights{weights}, m_biases{biases}, m_neuronsPerLayer{neuronsPerLayer},
                                    m_learningRate{learningRate}
{}

const std::vector<matrix<double>>& neural_network::get_weights() const
{
    return m_weights;
}

const std::vector<math_vector<double>>& neural_network::get_biases() const
{
    return m_biases;
}

const std::vector<std::size_t>& neural_network::get_neurons_per_layer() const
{
    return m_neuronsPerLayer;
}

double neural_network::get_learning_rate() const
{
    return m_learningRate;
}