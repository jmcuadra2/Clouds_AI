// File: NeuralNet.h
// Purpose: Header file for position inference using a GRU net

#ifndef GRUNET_H
#define GRUNET_H
#include "NeuralNet.h"

/**
    Class for position inference using a GRU neural network
*/

class GRUNet : public NeuralNet
{
public:
    GRUNet(int input_size, int output_size, int hidden_dim, int n_layers, float force, int sliding_window = 10);
    void forward(torch::Tensor input) override;

protected:
    torch::nn::GRU gru = nullptr;  // Torch multi-layer gated recurrent unit for calling forward pass 

};

#endif // GRUNET_H
