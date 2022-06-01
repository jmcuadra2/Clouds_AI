#include "CumulusIA.h"
#include <random>
#include <iostream>
#include <algorithm>
#include <GL/glu.h>

#include <chrono>
#include <math.h>

using namespace nimbus;

/**
 * @brief Constructor
 * 
 */
CumulusIA::CumulusIA() : Cumulus()
{  
    this->M = 30;
    this->N = 7;
    this->O = 7;
    
    windForce = 0.2; 
    coefDisp = 0.03;
    
    meanXPos = 0;
    meanYPos = 0;
    meanZPos = 0;  
    
   devXPos = 0;
   devYPos = 0;
   devZPos = 0;     
}

/**
 * @brief Create cumulus
 * 
 * @param spheres Number of spheres to generate cloud
 * @param siz Size of Gaussian cloud
 * @param center Cloud position
 * @param nuX Gaussian X-mean
 * @param sigX Gaussian X-standard deviation
 * @param nuY Gaussian Y-mean
 * @param sigY Gaussian Y-standard deviation
 * @param nuZ Gaussian Z-mean
 * @param sigZ Gaussian Z-standard deviation
 * @param isFlat Cloud with level of condensation
 * @param optimze Optimze cloud spheres
 */
void CumulusIA::create ( int spheres, GLfloat size, glm::vec3& center, GLfloat nuX, GLfloat sigX, GLfloat nuY, GLfloat sigY, GLfloat nuZ, GLfloat sigZ, bool isFlat, bool optimize )
{
    Cumulus::create(spheres, size, center, nuX, sigX, nuY, sigY, nuZ, sigZ, isFlat, optimize);   
    
    // JMCT Accelerated movement?
    dx = (float *)malloc(numSph * sizeof(float));
	dy = (float *)malloc(numSph * sizeof(float));
	dz = (float *)malloc(numSph * sizeof(float));
    for (int i = 0; i < numSph; i++) {
		dx[i] = dy[i] = dz[i] =  0.0f;
	}    
}

/**
 * @brief Destructor
 * 
 */
CumulusIA::~CumulusIA()
{
    if(dx) 
        free(dx);
    if(dy) 
        free(dy);
    if(dz) 
        free(dz);
}

/**
 * @brief Sets the neural network for the inference of the positions of the centers of the spheres
 * 
 * @param net Pointer to NeuraNet
 */
void CumulusIA::setNeuralNet(NeuralNet *net)
{
    this->net = net;
}

/**
 * @brief Sets 3D fluid grid sizes, wind force and dispersion control coefficient
 * 
 * @param M 3D fluid grid size, X axis
 * @param N 3D fluid grid size, Y axis
 * @param O 3D fluid grid size, Z axis
 * @param windForce Wind force
 * @param coefDisp Dispersion control coefficient
 */
void CumulusIA::setNetGridValues(int M, int N, int O, GLfloat windForce, GLfloat coefDisp/*, bool realistic*/)
{
    this->M = M;
    this->N = N;
    this->O = O;

    this->windForce = windForce;
    this->coefDisp = coefDisp; 
    
}

/**
 * @brief Sets positions of the centers of the cloud spheres and raduis
 * 
 * @param index Sphere index
 * @param position vector with positions coordinates, fourth coordinate is the radius
 */
void CumulusIA::setPosition(int index, glm::vec4 position)
{
    sphPos[index + lowLimit] = position;
}

/**
 * @brief Get inital inferences from neural network
 * 
 */
void CumulusIA::initialState()
{
    std::vector<glm::vec3> total_data;
    std::cout << "starting" << std::endl;
    int j_end = net->getSlidingWindow();
    for(int j = 0; j < j_end; j++) {
        std::vector<glm::vec3> vel_list;
        GLfloat auxU = 0;  // Avance lento en +X
        GLfloat auxV = 0; 
        GLfloat auxW = 0;
        GLfloat radius = 0;
        for(int i = 0; i < NeuralNet::getNUM_SPH(); i++) {
            if(i < numSph) {
                GLfloat x = sphPos[i + lowLimit].x - center.x;
                GLfloat y = sphPos[i + lowLimit].y - center.y;
                GLfloat z = sphPos[i + lowLimit].z - center.z;                               
                radius = sphPos[i + lowLimit].w;
  
                
                GLfloat px = x + M/2.0;
                GLfloat py = y + N/2.0;
                GLfloat pz = z + O/2.0;                
                
                // Maintains consistent values ​​within range
                px = glm::clamp(px,0.0f,(float)M) - ((double) rand() / (RAND_MAX + 1.))/10.;
                py = glm::clamp(py,0.0f,(float)N) - ((double) rand() / (RAND_MAX + 1.))/10.;
                pz = glm::clamp(pz,0.0f,(float)O) - ((double) rand() / (RAND_MAX + 1.))/10.;
                
                
                GLfloat uF = net->getUForce(j, px, py, pz);
                GLfloat vF = net->getVForce(j, px, py, pz);
                GLfloat wF = net->getWForce(j, px, py, pz);
                
                GLfloat random_y = ((double) rand() / (RAND_MAX + 1.)) * 0.06 - 0.03;
                GLfloat random_z = ((double) rand() / (RAND_MAX + 1.)) * 0.06 - 0.03;
                           

                auxU = sphPos[i + lowLimit].x + glm::clamp(uF, -0.5f, 1.0f) / 10.0;  // Slow advance on +X
                auxV = sphPos[i + lowLimit].y + glm::clamp(vF - random_y, -0.2f, 0.2f);
                auxW = sphPos[i + lowLimit].z + glm::clamp(wF - random_z - random_y, -0.2f, 0.2f);                

                vel_list.push_back(glm::vec3(uF, vF-random_y, wF-random_z)); 
                sphPos[i + lowLimit] = glm::vec4(auxU, auxV, auxW, radius);
            }
            else { // zero padding
                vel_list.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
            }
        }
        
        total_data.insert(total_data.end(), vel_list.begin(), vel_list.end());            
    }

    auto options = torch::TensorOptions();
    std::cout << "options " << options << std::endl;
    std::cout << "total_data " << total_data.size() << std::endl;
    
    inp = torch::from_blob(total_data.data(), {j_end, NeuralNet::getNUM_SPH(), 3}, options).view({1, -1, NeuralNet::getNUM_SPH() * 3}).toType(torch::kFloat).to(net->getDevice()).clone(); // view() in move_cloud() Python latten (3*35) y mete el batch dim el -1 equivale a 10 del sliding
    std::cout << "starting " << inp.sizes() << " " << inp.dtype() << std::endl; 
    std::cout << "starting " << inp << std::endl; 
}

/**
 * @brief Update positions of the centers of the cloud spheres and encompassing box
 * 
 */
void CumulusIA::updatePosition(/*float display_step*/) {
    
    glm::vec3 maxPos = glm::vec3(-999999999.0f);
	glm::vec3 minPos = glm::vec3(99999999.0f);
    
    if(numSph == 0)
        std::cout << "SPheres 0" << std::endl;

    for(int k = 0; k < numSph; k++) {       
        updatePos(k/*, display_step*/);
        if(maxPos.x < sphPos[k +  lowLimit].x + sphRads[k])
            maxPos.x = sphPos[k +  lowLimit].x + sphRads[k];
        if(maxPos.y < sphPos[k +  lowLimit].y + sphRads[k])
            maxPos.y = sphPos[k +  lowLimit].y + sphRads[k];
        if(maxPos.z < sphPos[k +  lowLimit].z + sphRads[k])
            maxPos.z = sphPos[k +  lowLimit].z + sphRads[k];
        if(minPos.x > sphPos[k +  lowLimit].x - sphRads[k])
            minPos.x = sphPos[k +  lowLimit].x - sphRads[k]; 
        if(minPos.y > sphPos[k +  lowLimit].y - sphRads[k])
            minPos.y = sphPos[k +  lowLimit].y - sphRads[k];
        if(minPos.z > sphPos[k +  lowLimit].z - sphRads[k])
            minPos.z = sphPos[k +  lowLimit].z - sphRads[k];        
    
    }

    vmax[id] = maxPos;
    vmin[id] = minPos;
    boxDimensions = maxPos - minPos;
    if(flat[id])
		vmin[id].y+=0.5f;
    
}

/**
 * @brief Update position of the center of one spheres
 * 
 * @param index Sphere index
 */
void CumulusIA::updatePos(int index/*, float display_step*/) {
    //JMCT  
    sphPos[index + lowLimit].x = sphPos[index + lowLimit].x + dx[index];

    sphPos[index + lowLimit].y = glm::clamp(sphPos[index + lowLimit].y + dy[index], meanYPos - 0.0f, meanYPos + 5.0f);
    sphPos[index + lowLimit].z = glm::clamp(sphPos[index + lowLimit].z + dz[index], meanZPos - 5.0f, meanZPos + 5.0f);

}

/**
 * @brief Calculate statistics of centers of spheres
 * 
 */
void CumulusIA::calculateMeans()
{
    for (int k = 0; k < numSph; k++) {
        meanXPos += sphPos[k + lowLimit].x;
        meanYPos += sphPos[k + lowLimit].y;
        meanZPos += sphPos[k + lowLimit].z;
        devXPos += sphPos[k + lowLimit].x * sphPos[k + lowLimit].x;
        devYPos += sphPos[k + lowLimit].y * sphPos[k + lowLimit].y;
        devZPos += sphPos[k + lowLimit].z * sphPos[k + lowLimit].z;
    }
    meanXPos = meanXPos / numSph;
    meanYPos = meanYPos / numSph;
    meanZPos = meanZPos / numSph; 
    devXPos = sqrt(devXPos / numSph - meanXPos * meanXPos);
    devXPos = sqrt(devYPos / numSph - meanYPos * meanYPos);
    devXPos = sqrt(devZPos / numSph - meanZPos * meanZPos); 

}

/**
 * @brief Get inferences from neural network and clamps them so that the cloud does not disintegrate
 * 
 */
void CumulusIA::inferPos()
{
    torch::NoGradGuard no_grad;
    
    auto start = std::chrono::steady_clock::now();
    
    auto start1 = std::chrono::steady_clock::now();
    net->forward(inp); 
    torch::Tensor& output = net->getOutput();    

    int fake_none = inp.sizes()[1];
    inp.index({Slice(), Slice(None, -1)}) = inp.index({Slice(), Slice(1, fake_none)}).clone().to(net->getDevice());
    inp.index({Slice(), -1}) = output.clone().to(net->getDevice());    

    auto end1 = std::chrono::steady_clock::now();   
    auto start2 = std::chrono::steady_clock::now();
    torch::Tensor output_cpu = output.cpu();

    auto end2 = std::chrono::steady_clock::now();
    
    auto start3 = std::chrono::steady_clock::now();   
    meanXinp_out3 = 0;
    meanYinp_out3 = 0;
    meanZinp_out3 = 0;
    int cnt = 0;
    
    auto accessor = output_cpu.accessor<float,2>();
    int o_size = output.size(1);
    for(int i = 0; i < o_size; i += 3) {
        meanXinp_out3 += accessor[0][i];
        meanYinp_out3 += accessor[0][i + 1];
        meanZinp_out3 += accessor[0][i + 2];
        ++cnt;
    }
    meanXinp_out3 = meanXinp_out3 / cnt;
    meanYinp_out3 = meanYinp_out3 / cnt;
    meanZinp_out3 = meanZinp_out3 / cnt;        
    
    for (int k = 0; k < numSph; k++) {
        dx[k] = glm::clamp(meanXinp_out3 + coefDisp * (accessor[0][k*3] - meanXinp_out3), 0.1f, 2.0f);
        dy[k] = accessor[0][k * 3 + 1];
        dz[k] = accessor[0][k*3 + 2];           
    }
    
  
}
