// File: Main.h
// Purpose: Main application header file

// Required headers

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#include <stdlib.h> 
// JMCT
#ifdef _MSC_VER
#include <crtdbg.h>
#else
#define _ASSERT(expr) ((void)0)
#define _ASSERTE(expr) ((void)0)
#endif 

#include <GL/glew.h>
// JMCT
#ifdef _WIN32
#include <GL/wglew.h> // For wglSwapInterval
#elif __linux__
#include <GL/glxew.h> 
#include <X11/Xlib.h>
#endif

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
#include "Lsystem.h"
#include "Metacloud.h"
#include "Exception.h"
