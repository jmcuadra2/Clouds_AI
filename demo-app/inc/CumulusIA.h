// File: CumulusIA.h
// Purpose: Header file for cumulus rendering using a neural network for position inference

#ifndef CUMULUSIA_H
#define CUMULUSIA_H

#include <glm.hpp>
#include "NeuralNet.h"
#include "Cumulus.h"
#include <GL/gl.h>
#include <vector>

/**
 * @todo write docs
 */
namespace nimbus
{
	/**
	Class for cumulus rendering using a neural network for position inference
	*/    
    class CumulusIA : public Cumulus
    {
    public:

        CumulusIA();

        /**
        * Destructor
        */
        ~CumulusIA();
        
        void create ( int spheres, GLfloat size, glm::vec3& center, GLfloat nuX, GLfloat sigX, GLfloat nuY, GLfloat sigY, GLfloat nuZ, GLfloat sigZ, bool isFlat, bool optimize ) override;

        void inferPos();
        void setNeuralNet(NeuralNet *net);
        void setNetGridValues(int M, int N, int O, GLfloat windForce, GLfloat coefDisp/*, bool realistic = false*/);
        void initialState();
        void setPosition(int index, glm::vec4 position);    
        void updatePosition(/*float display_step = 1.0*/);
        void calculateMeans();     
        
    protected:    
        virtual void updatePos(int index/*, float display_step = 1.0*/);

        inline int IX(GLfloat i, GLfloat j, GLfloat k)
            {
                return i+(M+2)*(j) + (M+2)*(N+2)*(k);
            }
            
    protected:
        glm::vec3 center;
        
        GLfloat windForce; // Wind force (values between 0 and 0.3)
        GLfloat coefDisp;  // Dispersion control coefficient, the higher the more it disintegrates when advancing (values ​​between 0 and 0.1 typically)
        
        
        GLfloat meanXinp_out3; // mean of every 3 inference outputs, X coordinate
        GLfloat meanYinp_out3; // mean of every 3 inference outputs, Y coordinate
        GLfloat meanZinp_out3; // mean of every 3 inference outputs, Z coordinate
        GLfloat meanXPos; // mean positions of the centers of the cloud spheres, X coordinate
        GLfloat meanYPos; // mean positions of the centers of the cloud spheres, Y coordinate
        GLfloat meanZPos; // mean positions of the centers of the cloud spheres, Z coordinate 
        GLfloat devXPos; // standard deviation of the positions of the centers of the cloud spheres, X coordinate
        GLfloat devYPos;// standard deviation of the positions of the centers of the cloud spheres, Y coordinate
        GLfloat devZPos;// standard deviation of the positions of the centers of the cloud spheres, Z coordinate         
        
        float *dx, *dy, *dz; // vectors with the coordinates of the displacements of the centers of cloud spheres
        glm::vec3 boxDimensions; // cloud encompassing box
        
        NeuralNet *net; // Pointer to neuran net object
        torch::Tensor inp; // forward pass output

//         bool realistic;
        
    private:
        int M; // 3D fluid grid size, X axis
        int N; // 3D fluid grid size, Y axis
        int O; // 3D fluid grid size, Z axis            
    };
}

#endif // CUMULUSIA_H

