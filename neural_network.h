//
// Created by cartercpp on 8/7/26.
//

#ifndef DIGITCLASSIFICATION_NEURAL_NETWORK_H
#define DIGITCLASSIFICATION_NEURAL_NETWORK_H

#include <initializer_list>
#include <vector>
#include <cstddef>
#include "math_vector.h"
#include "matrix.h"

class neural_network
{
public:

    // CONSTRUCTORS

    explicit neural_network(std::initializer_list<std::size_t>, double);

    // METHODS

    math_vector<double> predict(const math_vector<double>&) const;
    void fit(const math_vector<double>&, const math_vector<double>&);

private:

    static double Relu(double) noexcept;
    static double ReluDerivative(double) noexcept;

    std::vector<math_vector<double>> Forward(const math_vector<double>&) const;

    std::vector<matrix<double>> m_weights;
    std::vector<math_vector<double>> m_biases;
    std::vector<std::size_t> m_neuronsPerLayer;
    double m_learningRate;
};

#endif //DIGITCLASSIFICATION_NEURAL_NETWORK_H