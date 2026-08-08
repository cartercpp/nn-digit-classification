#include <iostream>
#include <string>
#include <array>
#include <print>
#include <algorithm>
#include <random>
#include <thread>
#include <stop_token>
#include <chrono>
#include <cstddef>
#include "math_vector.h"
#include "neural_network.h"

extern const std::array<std::string, 10> digits;

int main()
{
    std::random_device rd;
    std::uniform_int_distribution<int> dist(0, 9);

    neural_network digitNN({784, 512, 256, 10}, 0.01);
    math_vector<double> vec(784);

    auto moveCursor = [](int row, int column)
    {
        std::print("\033[{};{}H", row + 1, column + 1);
    };

    std::print("\033[?25l"); // cursor off
    std::print("\033[2J"); // clear screen
    moveCursor(0, 0);
    std::println("Training progress: 0%");

    constexpr int trainingEpochs = 100'000;
    for (int epoch = 1; epoch <= trainingEpochs; ++epoch)
    {
        const int digit = dist(rd);
        for (int index = 0; index < 784; ++index)
            vec[index] = (digits[digit][index] != ' ');

        math_vector<double> target(10, 0);
        target[digit] = 1;
        digitNN.fit(vec, target);

        if (epoch % (trainingEpochs / 10) == 0)
            std::println("Training progress: {}%", epoch * 100 / trainingEpochs);
    }

    std::cin.get();
    std::print("\033[2J"); // clear screen

    std::jthread thr{[&](std::stop_token st)
    {
        while (!st.stop_requested())
        {
            const int digit = dist(rd);

            moveCursor(0, 0);
            for (int row = 0; row < 28; ++row)
            {
                for (int column = 0; column < 28; ++column)
                {
                    const int index = row * 28 + column;
                    std::cout << digits[digit][index];
                    vec[index] = (digits[digit][index] != ' ');
                }
                std::cout << '\n';
            }

            const auto classification = digitNN.predict(vec);
            int predictedDigit = 0;
            for (int i = 1; i < 10; ++i)
                if (classification[i] > classification[predictedDigit])
                    predictedDigit = i;

            std::print("Prediction: {} ({:.2f}% confidence)", predictedDigit, classification[predictedDigit] * 100);

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }};
    std::cin.get();
}