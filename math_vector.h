//
// Created by cartercpp on 8/7/26.
//

#ifndef DIGITCLASSIFICATION_MATH_VECTOR_H
#define DIGITCLASSIFICATION_MATH_VECTOR_H

#include <stdexcept>
#include <initializer_list>
#include <vector>
#include <algorithm>
#include <concepts>
#include <cstddef>

template <typename ValueType> requires (std::integral<ValueType> || std::floating_point<ValueType>)
class math_vector
{
public:

    // CONSTRUCTORS

    explicit math_vector(std::size_t size, ValueType value = 0)
        : m_data(size, value)
    {}

    math_vector(std::initializer_list<ValueType> data)
        : m_data{data}
    {}

    // METHODS

    math_vector multiply(const math_vector& other) const
    {
        if (m_data.size() != other.size())
            throw std::invalid_argument{"In order to multiply 2 matrices, they must be the same size"};

        math_vector<ValueType> output(m_data.size());
        for (std::size_t i = 0; i < m_data.size(); ++i)
            output[i] = m_data[i] * other[i];

        return output;
    }

    math_vector& operator*=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](ValueType& valueRef){valueRef *= scalar;});
        return *this;
    }

    math_vector& operator/=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](ValueType& valueRef){valueRef /= scalar;});
        return *this;
    }

    math_vector& operator+=(const math_vector& other)
    {
        if (other.size() != m_data.size())
            throw std::invalid_argument{"In order to add 2 vectors, they must be the same size"};

        for (std::size_t i = 0; i < m_data.size(); ++i)
            m_data[i] += other[i];

        return *this;
    }

    math_vector& operator-=(const math_vector& other)
    {
        if (other.size() != m_data.size())
            throw std::invalid_argument{"In order to add 2 vectors, they must be the same size"};

        for (std::size_t i = 0; i < m_data.size(); ++i)
            m_data[i] -= other[i];

        return *this;
    }

    ValueType& operator[](std::size_t index)
    {
        if (index >= m_data.size())
            throw std::out_of_range{"Out of bounds"};

        return m_data[index];
    }

    const ValueType& operator[](std::size_t index) const
    {
        if (index >= m_data.size())
            throw std::out_of_range{"Out of bounds"};

        return m_data[index];
    }

    std::size_t size() const
    {
        return m_data.size();
    }

private:

    std::vector<ValueType> m_data;
};

template <typename ValueType>
auto operator*(math_vector<ValueType> vec, ValueType scalar)
{
    return vec *= scalar;
}

template <typename ValueType>
auto operator*(ValueType scalar, math_vector<ValueType> vec)
{
    return vec *= scalar;
}

template <typename ValueType>
auto operator/(math_vector<ValueType> vec, ValueType scalar)
{
    return vec /= scalar;
}

template <typename ValueType>
auto operator+(math_vector<ValueType> lArg, const math_vector<ValueType>& rArg)
{
    return lArg += rArg;
}

template <typename ValueType>
auto operator-(math_vector<ValueType> lArg, const math_vector<ValueType>& rArg)
{
    return lArg -= rArg;
}

template <typename ValueType>
auto operator*(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
{
    ValueType output = 0;
    for (std::size_t i = 0; i < lArg.size(); ++i)
        output += lArg[i] * rArg[i];

    return output;
}

#endif //DIGITCLASSIFICATION_MATH_VECTOR_H