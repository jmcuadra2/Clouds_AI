// File: NeuralNet.h
// Purpose: Header file for position inference using a neural network

#ifndef NEURALNET_H
#define NEURALNET_H

#include "FluidCPU.h"
#include <torch/torch.h>

using namespace torch::indexing;

	/**
	Base class for position inference using a neural network
	*/

class NeuralNet : public torch::nn::Module
{
public:
    NeuralNet (int input_size, int output_size, int hidden_dim, int n_layers, float force, int sliding_window = 10);
    virtual ~NeuralNet();

    virtual void forward ( torch::Tensor input);
    int getSlidingWindow();
    torch::Device& getDevice();
    void loadWeigths(std::string path);
    torch::Tensor& getOutput();
    bool setFluidData(int M, int N, int O, float dt, float diff, float visc);
    void initWindGrid();
    inline float getUForce(int s, int i, int j, int k)
    {
        return u[IX(s, i, j, k)];
    }
    inline float getVForce(int s, int i, int j, int k)
    {
        return v[IX(s, i, j, k)];
    }
    inline float getWForce(int s, int i, int j, int k)
    {
        return w[IX(s, i, j, k)];
    }
    int getM();
    int getN();
    int getO();
        
    static void setNUM_SPH(int num_sph);
    static int getNUM_SPH();
    static void setDeviceName(std::string device_name);

protected:
    void initLayer ( torch::Tensor& layer );
    
    bool allocateGridData();
    void initGridData();
    
    /**
     * @brief Fast index retrieval
     * 
     * @param s Sliding window index
     * @param i i-index
     * @param j j-index
     * @param k k-index
     * @return int
     */
    inline int IX(int s, int i, int j, int k)
    {
        return (int)(i + (M+2)*j + (M+2)*(N+2)*k + (M+2)*(N+2)*(O+2)*s);
    }

protected:
    int inputSize; // Number of neurons in the input layer
    int outputSize; // Number of neurons in the output layer
    int hiddenDim; // Number of neurons in the hidden layers
    int numLayers; // Number of hidden layers
    int batchSize; // Training batch size
    float windForce; // Wind force
    int slidingWindow; // Sliding window size
    
    int M; // Dimension of fluid simulator, X
    int N; // Dimension of fluid simulator, Y
    int O; // Dimension of fluid simulator, Z

    torch::Tensor cell; // Cell state tensor in LSMT
    torch::Tensor hidden; // Hidden state tensor
    torch::Tensor output; // Output state tensor

    torch::Device device; // Torch device, CPU or CUDA

    torch::nn::Linear fc = nullptr; // Fully conected layer
    
    nimbus::FluidCPU* windGrid; // Pointer to wind 3D grid
    float *u, *v, *w;
    
private:
    static int NUM_SPH; // number of spheres in training
    static std::string deviceName; // torch device CPU or CUDA

};

#endif // NEURALNET_H
