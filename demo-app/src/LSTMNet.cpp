#include "LSTMNet.h"

#include <chrono>

/**
 * @brief Neural network LSTM constructor
 * 
 * @param input_size Number of neurons in the input layer
 * @param output_size Number of neurons in the output layer
 * @param hidden_dim Number of neurons in the hidden layers
 * @param n_layers Number of hidden layers
 * @param force Wind force
 * @param sliding_window Sliding window size
 */
LSTMNet::LSTMNet(int input_size, int output_size, int hidden_dim, int n_layers, float force, int sliding_window) : NeuralNet(input_size, output_size, hidden_dim, n_layers, force, sliding_window), lstm(register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_dim).num_layers(n_layers).batch_first(true))))
{
   
}

/**
 * @brief Execute neural network forward pass
 * 
 * @param input Input tensor values
 */
void LSTMNet::forward(at::Tensor input)
{
    torch::NoGradGuard no_grad;
    NeuralNet::forward(input);
    initLayer(hidden);
    initLayer(cell);

    std::tuple<torch::Tensor, torch::Tensor> hx_opt;    
    torch::Tensor output1;
        
    std::tie(output1, hx_opt) = lstm->forward(input, std::make_tuple(hidden, cell));
    hidden = std::get<0>(hx_opt).to(device);
    cell = std::get<1>(hx_opt).to(device);

    torch::Tensor output2 = output1.index({Slice(), -1}).to(device);
    output = fc(output2).to(device);

}
