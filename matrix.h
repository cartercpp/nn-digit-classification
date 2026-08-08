//
// Created by cartercpp on 8/7/26.
//

#ifndef DIGITCLASSIFICATION_MATRIX_H
#define DIGITCLASSIFICATION_MATRIX_H

#include <stdexcept>
#include <initializer_list>
#include <vector>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include "math_vector.h"

template <typename ValueType> requires (std::integral<ValueType> || std::floating_point<ValueType>)
class matrix
{
public:

    // CONSTRUCTORS

    explicit matrix(std::size_t rows, std::size_t columns, ValueType value = 0)
        : m_data(rows, math_vector<ValueType>(columns, value)), m_rows{rows}, m_columns{columns}
    {}

    explicit matrix(std::size_t rows, const math_vector<ValueType>& vec)
        : m_data(rows, vec), m_rows{rows}, m_columns{vec.size()}
    {}

    matrix(std::initializer_list<math_vector<ValueType>> data)
        : m_data{data}, m_rows{data.size()}, m_columns{(*data.begin()).size()}
    {}

    // METHODS

    matrix transpose() const
    {
        matrix<ValueType> output(m_columns, m_rows);

        for (std::size_t row = 0; row < m_rows; ++row)
            for (std::size_t column = 0; column < m_columns; ++column)
                output[column][row] = m_data[row][column];

        return output;
    }

    matrix& operator*=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](math_vector<ValueType>& vecRef){vecRef *= scalar;});
        return *this;
    }

    matrix& operator/=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](math_vector<ValueType>& vecRef){vecRef /= scalar;});
        return *this;
    }

    matrix& operator+=(const matrix& other)
    {
        if ((m_rows != other.rows()) || (m_columns != other.columns()))
            throw std::invalid_argument{"In order to add 2 matrices, they must be the same size"};

        for (std::size_t row = 0; row < m_rows; ++row)
            m_data[row] += other[row];

        return *this;
    }

    matrix& operator-=(const matrix& other)
    {
        if ((m_rows != other.rows()) || (m_columns != other.columns()))
            throw std::invalid_argument{"In order to add 2 matrices, they must be the same size"};

        for (std::size_t row = 0; row < m_rows; ++row)
            m_data[row] -= other[row];

        return *this;
    }

    math_vector<ValueType>& operator[](std::size_t row)
    {
        if (row >= m_data.size())
            throw std::out_of_range{"Out of bounds"};

        return m_data[row];
    }

    const math_vector<ValueType>& operator[](std::size_t row) const
    {
        if (row >= m_data.size())
            throw std::out_of_range{"Out of bounds"};

        return m_data[row];
    }

    std::size_t rows() const
    {
        return m_rows;
    }

    std::size_t columns() const
    {
        return m_columns;
    }

private:

    std::vector<math_vector<ValueType>> m_data;
    std::size_t m_rows,
                m_columns;
};

template <typename ValueType>
auto outer_product(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
{
    matrix<ValueType> output(lArg.size(), rArg.size());

    for (std::size_t i = 0; i < lArg.size(); ++i)
        for (std::size_t i2 = 0; i2 < rArg.size(); ++i2)
            output[i][i2] = lArg[i] * rArg[i2];

    return output;
}

template <typename ValueType>
auto operator*(const matrix<ValueType>& mat, const math_vector<ValueType>& vec)
{
    if (mat.columns() != vec.size())
        throw std::invalid_argument{"In order to perform a matrix-vector multiplication, the matrix must have as many"
                                    " columns as the vector has rows"};

    math_vector<ValueType> output(mat.rows());
    for (std::size_t i = 0; i < mat.rows(); ++i)
        output[i] = mat[i] * vec;

    return output;
}

template <typename ValueType>
auto operator*(matrix<ValueType> mat, ValueType scalar)
{
    return mat *= scalar;
}

template <typename ValueType>
auto operator*(ValueType scalar, matrix<ValueType> mat)
{
    return mat *= scalar;
}

template <typename ValueType>
auto operator/(matrix<ValueType> mat, ValueType scalar)
{
    return mat /= scalar;
}

template <typename ValueType>
auto operator+(matrix<ValueType> lArg, const matrix<ValueType>& rArg)
{
    return lArg += rArg;
}

template <typename ValueType>
auto operator-(matrix<ValueType> lArg, const matrix<ValueType>& rArg)
{
    return lArg -= rArg;
}

#endif //DIGITCLASSIFICATION_MATRIX_H