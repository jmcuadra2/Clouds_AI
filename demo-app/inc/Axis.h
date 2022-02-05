// File: Axis.h
// Purpose: Header file for 3D-Axis rendering


#pragma once

#include <GL/glew.h>
// JMCT
#ifdef _WIN32
#include <GL/wglew.h> // For wglSwapInterval
#elif __linux__
#include <GL/glxew.h> 
#endif

#include <GL/freeglut.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include "Camera.h"
#include "Shader.h" 

namespace nimbus
{
	/**
	Class for 3D-Axis rendering
	*/
	class Axis
	{
	private:
		GLuint vertexbufferShaderAxis;
		GLint iUniformMVPAxis; // Uniforms
		GLint iLineColor;
		// Buffer for axis 3-lines (X,Y,Z)
		GLfloat vertexBufferDataAxis[6];
	public:
		Axis();
		void create();
		void getUniforms(Shader& shader);
		void render(Camera& cameraAxis);
		~Axis();
	};
}

