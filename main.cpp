#include <iostream>
#include <string>
#include <array>
#include <print>
#include <algorithm>
#include <random>
#include <thread>
#include <stop_token>
#include <chrono>
#include "math_vector.h"
#include "neural_network.h"

extern const std::array<std::string, 10> digits;
extern neural_network digitNN;

int main()
{
    std::random_device rd;
    std::uniform_int_distribution<int> dist(0, 9);

    math_vector<double> vec(784);

    auto moveCursor = [](int row, int column)
    {
        std::print("\033[{};{}H", row + 1, column + 1);
    };

    std::print("\033[?25l"); // cursor off
    std::print("\033[2J"); // clear screen
    moveCursor(0, 0);

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
            std::cout << std::flush;

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }};
    std::cin.get();
    std::print("\033[?25h"); // restore cursor
}