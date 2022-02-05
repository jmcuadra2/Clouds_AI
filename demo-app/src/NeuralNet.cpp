#include "NeuralNet.h"

int NeuralNet::NUM_SPH = 0;
std::string NeuralNet::deviceName = "cuda";

/**
 * @brief Sets number of spheres
 * 
 * @param num_sph Number of spheres
 */
void NeuralNet::setNUM_SPH(int num_sph)
{
    NeuralNet::NUM_SPH = num_sph;
}

/**
 * @brief Returns number of spheres
 * 
 * @return int Number of spheres
 */
int NeuralNet::getNUM_SPH()
{
    return NeuralNet::NUM_SPH;
}

/**
 * @brief Sets torch device, CPU or CUDA
 * 
 * @param device_name Torch device
 */
void NeuralNet::setDeviceName(std::string device_name)
{
    NeuralNet::deviceName = device_name;
}

/**
 * @brief Neural network constructor
 * 
 * @param input_size Number of neurons in the input layer
 * @param output_size Number of neurons in the output layer
 * @param hidden_dim Number of neurons in the hidden layers
 * @param n_layers Number of hidden layers
 * @param force Wind force
 * @param sliding_window Sliding window size
 */
NeuralNet::NeuralNet(int input_size, int output_size, int hidden_dim, int n_layers, float force, int sliding_window) : device(torch::Device(deviceName)), fc(register_module("fc", torch::nn::Linear(hidden_dim, input_size)))
{
    inputSize = input_size;
    outputSize = output_size;
    hiddenDim = hidden_dim;
    numLayers = n_layers;
    windForce = force;
    slidingWindow = sliding_window;    
}

/**
 * @brief Neural network destructor
 * 
 */
NeuralNet:: ~NeuralNet() 
{
   delete windGrid; 
}

/**
 * @brief Set the parameters for FluidCPU constructor
 * 
 * @param M Grid size X.
 * @param N Grid size Y
 * @param O Grid size Z
 * @param dt Time delta
 * @param diff Diffuse
 * @param visc Viscosity.
 * @return bool True if grid data can be allocated
 */
bool NeuralNet::setFluidData(int M, int N, int O, float dt, float diff, float visc)
{
    this->M = M;
	this->N = N;
	this->O = O;
    windGrid = new nimbus::FluidCPU(M, N, O, dt, diff, visc);
    if (!allocateGridData())
		return false;
	initGridData();
    return true;
}

/**
 * @brief Load neural network wights from .pt file
 * 
 * @param path p_path:...
 */
void NeuralNet::loadWeigths(std::string path)
{
//     try {
    torch::serialize::InputArchive archive;
    archive.load_from(path, device);
    for (const auto& key : archive.keys()) {
          std::cout << key << std::endl;
    }
    load(archive);
    

    eval();
    to(device);
    std::cout << "Usando el device " << device << std::endl;

}

/**
 * @brief Execute neural network forward pass in subclasses
 * 
 * @param input Input tensor values
 */
void NeuralNet::forward(torch::Tensor input)
{
    batchSize = input.size(0);
}

/**
 * @brief Initialze a neural network layer
 * 
 * @param layer Neural network layer
 */
void NeuralNet::initLayer(torch::Tensor& layer)
{
    c10::IntArrayRef sizes = layer.sizes();
//     std::cout << "layer sizes " << sizes << std::endl;
    if(sizes[0] == 0) { // Just create it at the start
        layer = torch::zeros({numLayers, batchSize, hiddenDim}).to(torch::kFloat32).to(device);
    }
//     std::cout << "layer device " << layer.device() << std::endl;
}

/**
 * @brief Initialize array pointers for 3D fluid components with wind forces
 * 
 */
void NeuralNet::initWindGrid()
{
    // s = 1 to leave the first slide with the 0's
    for(int s = 1; s < slidingWindow; s++) {
        windGrid->setPrevForceSource(windForce);    
        windGrid->sim();
        for(int i = 0; i < M + 2; i++) {
            for(int j = 0; j < N + 2; j++) {
                for(int k = 0; k < O + 2; k++) {
                    u[IX(s, i, j, k)] = windGrid->getUForce(i, j , k);
                    v[IX(s, i, j, k)] = windGrid->getVForce(i, j , k);
                    w[IX(s, i, j, k)] = windGrid->getWForce(i, j , k);               
                }                
            }
        }

    }  
}

/**
 * @brief Allocate memory for array pointers for 3D fluid components
 * 
 * @return bool True if grid data can be allocated
 */
bool NeuralNet::allocateGridData()
{
	int size = slidingWindow*(M + 2)*(N + 2)*(O + 2);

	u = (float *)malloc(size * sizeof(float));
	v = (float *)malloc(size * sizeof(float));
	w = (float *)malloc(size * sizeof(float));


	if (!u || !v || !w) {
		fprintf(stderr, "cannot allocate grid data\n");
		return false;
	}

	return true;
}

/**
 * @brief Initialize array pointers for 3D fluid components to 0
 * 
 */
void NeuralNet::initGridData()
{
	int size = slidingWindow*(M + 2)*(N + 2)*(O + 2);

	for (int i = 0; i < size; i++) {
		u[i] = v[i] = w[i] = 0.0f;
	}

}

/**
 * @brief Return sliding window size
 * 
 * @return int Ssliding window size
 */
int NeuralNet::getSlidingWindow() 
{
    return slidingWindow;
}

/**
 * @brief Returns torch device CPU or CUDA
 * 
 * @return c10::Device& Torch device
 */
torch::Device& NeuralNet::getDevice()
{
    return device;
}

/**
 * @brief Return neural network output values
 * 
 * @return at::Tensor& Neural network output values
 */
torch::Tensor& NeuralNet::getOutput()
{
    return output;
}

/**
 * @brief Return dimension of fluid simulator, X
 * 
 * @return int Dimension of fluid simulator, X
 */
int NeuralNet::getM()
{
    return M;
}

/**
 * @brief Return dimension of fluid simulator, Y
 * 
 * @return int Dimension of fluid simulator, Y
 */
int NeuralNet::getN()
{
    return N;
}

/**
 * @brief Return dimension of fluid simulator, Z
 * 
 * @return int Dimension of fluid simulator, Z
 */
int NeuralNet::getO()
{
    return O;
}
