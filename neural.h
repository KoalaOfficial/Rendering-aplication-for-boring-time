#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <cassert>

// Prosta warstwa sztucznej sieci neuronowej
class Neuron {
public:
    std::vector<double> weights;
    double bias;
    double output;
    double delta;

    Neuron(size_t input_size) {
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        weights.resize(input_size);
        for (auto &w : weights) w = dist(gen);
        bias = dist(gen);
    }

    static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    static double d_sigmoid(double x) { return x * (1.0 - x); }

    double activate(const std::vector<double> &inputs) {
        double sum = bias;
        for (size_t i = 0; i < weights.size(); ++i)
            sum += weights[i] * inputs[i];
        output = sigmoid(sum);
        return output;
    }
};

// Warstwa sieci
class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(size_t num_neurons, size_t input_size) {
        for (size_t i = 0; i < num_neurons; ++i)
            neurons.emplace_back(input_size);
    }

    std::vector<double> feed_forward(const std::vector<double> &inputs) {
        std::vector<double> outputs;
        for (auto &neuron : neurons)
            outputs.push_back(neuron.activate(inputs));
        return outputs;
    }
};

// Prosta sieć neuronowa
class NeuralNetwork {
public:
    std::vector<Layer> layers;

    NeuralNetwork(const std::vector<size_t> &layer_sizes) {
        assert(layer_sizes.size() >= 2);
        for (size_t i = 1; i < layer_sizes.size(); ++i)
            layers.emplace_back(layer_sizes[i], layer_sizes[i-1]);
    }

    std::vector<double> predict(const std::vector<double> &input) {
        std::vector<double> out = input;
        for (auto &layer : layers)
            out = layer.feed_forward(out);
        return out;
    }

    // Prosty algorytm uczenia (backpropagation)
    void train(const std::vector<std::vector<double>> &inputs,
               const std::vector<std::vector<double>> &targets,
               double lr = 0.1, size_t epochs = 1000) {
        for (size_t e = 0; e < epochs; ++e) {
            for (size_t sample = 0; sample < inputs.size(); ++sample) {
                // Forward
                std::vector<std::vector<double>> layer_outputs;
                layer_outputs.push_back(inputs[sample]);
                for (auto &layer : layers)
                    layer_outputs.push_back(layer.feed_forward(layer_outputs.back()));

                // Backward
                // Output layer error
                auto &out_layer = layers.back();
                for (size_t i = 0; i < out_layer.neurons.size(); ++i) {
                    double error = targets[sample][i] - out_layer.neurons[i].output;
                    out_layer.neurons[i].delta = error * Neuron::d_sigmoid(out_layer.neurons[i].output);
                }

                // Hidden layers error
                for (int l = layers.size() - 2; l >= 0; --l) {
                    auto &layer = layers[l];
                    auto &next_layer = layers[l+1];
                    for (size_t i = 0; i < layer.neurons.size(); ++i) {
                        double error = 0.0;
                        for (auto &next_neuron : next_layer.neurons)
                            error += next_neuron.weights[i] * next_neuron.delta;
                        layer.neurons[i].delta = error * Neuron::d_sigmoid(layer.neurons[i].output);
                    }
                }

                // Update weights & biases
                for (size_t l = 0; l < layers.size(); ++l) {
                    auto &layer = layers[l];
                    auto &inputs_ = layer_outputs[l];
                    for (auto &neuron : layer.neurons) {
                        for (size_t w = 0; w < neuron.weights.size(); ++w)
                            neuron.weights[w] += lr * neuron.delta * inputs_[w];
                        neuron.bias += lr * neuron.delta;
                    }
                }
            }
        }
    }
};
