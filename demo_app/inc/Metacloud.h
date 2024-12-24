// File: Metacloud.h
// Purpose: Header file for metaballs based clouds

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

#include "Shader.h"
#include "Cumulus.h"

namespace nimbus
{
	/**
	Class for metaballs based clouds rendering
	*/
	class Metacloud : public Cumulus
	{
	private:
		static GLint iMean; // Uniform
		static GLfloat mean[MAXCLOUDS]; // Mean of each cloud system
	public:
		Metacloud();
		static void getUniforms(Shader& shader);
		void create(int spheres, GLfloat siz, glm::vec3& center, GLfloat nuX, GLfloat sigX, GLfloat nuY, GLfloat sigY, GLfloat nuZ, GLfloat sigZ, bool isFlat, bool optimize);
		float getMean(int i);
		static void render();
	};
}
