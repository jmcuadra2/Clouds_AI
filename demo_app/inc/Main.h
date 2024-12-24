// File: Main.h
// Purpose: Main application header file

// Required headers

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#include <stdlib.h>  
#include <crtdbg.h>  

#include <GL/glew.h>
#include <GL/wglew.h> // For wglSwapInterval

#include <GL/freeglut.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

// Nimbus system inclusion

#include "Shader.h"
#include "Canvas.h"
#include "Mountain.h"
#include "Axis.h"
#include "Cloud.h"
#include "Cumulus.h"
#include "Model.h"
#include "Morph.h"
#include "LSystem.h"
#include "Metacloud.h"
#include "Exception.h"