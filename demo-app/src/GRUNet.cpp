#include "GRUNet.h"

/**
 * @brief Neural network GRU constructor
 * 
 * @param input_size Number of neurons in the input layer
 * @param output_size Number of neurons in the output layer
 * @param hidden_dim Number of neurons in the hidden layers
 * @param n_layers Number of hidden layers
 * @param force Wind force
 * @param sliding_window Sliding window size
 */
GRUNet::GRUNet(int input_size, int output_size, int hidden_dim, int n_layers, float force, int sliding_window) : NeuralNet(input_size, output_size, hidden_dim, n_layers, force, sliding_window), gru(register_module("gru", torch::nn::GRU(torch::nn::GRUOptions(input_size, hidden_dim).num_layers(n_layers).batch_first(true))))
{
    
}

/**
 * @brief Execute neural network forward pass
 * 
 * @param input Input tensor values
 */
void GRUNet::forward(at::Tensor input)
{
    torch::NoGradGuard no_grad;
    NeuralNet::forward(input);
    initLayer(hidden);

    torch::Tensor hx_opt;    
    torch::Tensor output1;
//     std::cout << "input " << input.sizes() << " "  << input.device() << std::endl;
//     std::cout << "hidden " << hidden.sizes() << " " << hidden.device() << std::endl;
        
    std::tie(output1, hx_opt) = gru->forward(input, hidden);
    hidden = hx_opt.to(device);

//     to(device) da problemas de tensor en dos devices)
//     std::cout << "output1 " << output1.sizes() << output1.dtype() << std::endl;
    torch::Tensor output2 = output1.index({Slice(), -1}).to(device);
//     std::cout << "output2 " << output2.sizes() << output2.dtype() << std::endl;
//     std::cout << "output2 " << output2 << std::endl;
    output = fc(output2).to(device);
//     std::cout << "output " << output.sizes() << output.dtype() << std::endl;
//     std::cout << "output " << output << std::endl;
}

