// File: NeuralNet.h
// Purpose: Header file for position inference using a LSTM net

#ifndef LSTMNET_H
#define LSTMNET_H
#include "NeuralNet.h"

	/**
	Class for position inference using a LSTM neural network
	*/

class LSTMNet : public NeuralNet
{
public:
    LSTMNet(int input_size, int output_size, int hidden_dim, int n_layers, float force, int sliding_window = 10);
    void forward(torch::Tensor input) override;

protected:
    torch::nn::LSTM lstm = nullptr; // Torch multi-layer long short term memory for calling forward pass 

};

#endif // LSTMNET_H
