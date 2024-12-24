// File: Shader.h
// Purpose: Header file for OpenGL shader handling

#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>

#include <GL/glew.h>
#include <GL/wglew.h> // For wglSwapInterval

#include <GL/freeglut.h>

#include "Exception.h"

namespace nimbus
{

	/**
	Class for OpenGL shader handling
	*/
	class Shader
	{
	private:
		GLuint shaderProgram;
		std::vector<GLuint> shadersList;
	public:
		Shader();
		void loadShader(GLenum shaderType, const std::string& shaderFile);
		void createShaderProgram();
		GLint getUniformLocation(const GLchar *name);
		GLuint getUniformBlockIndex(const GLchar *uniformBlockName);
		GLuint getProgram();
		void useProgram();
		~Shader();
	};
}

