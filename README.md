# NN Digit Classification

A from-scratch neural network for handwritten digit classification (0–9) written in modern C++23.

The network takes a 28×28 ASCII digit as input and predicts the class with a confidence score.

## Features

- Pure C++ implementation (no external ML libraries)
- Pre-trained feedforward neural network
- Custom `matrix` and `math_vector` classes
- Real-time terminal demo

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

Requires a C++23 compiler.

## Run

```bash
./DigitClassification
```

Press **Enter** to stop.

The program randomly displays ASCII digits and shows the network’s prediction along with its confidence.
