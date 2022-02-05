////////////////////////////////////////////////////////////////////
// ONTOGENETIC MODEL FOR REAL-TIME VOLUMETRIC CLOUDS SIMULATION THESIS			
// RECURRENT NEURAL NETWORK CLOUD DYNAMICS SIMULATION VERSION (PyTorch required)
// Software Engineering and Computer Systems Deparment	
// National University for Distance Education (UNED)			    		
// (c) Carlos Jiménez de Parga, PhD student.
// Last revision 20/01/2022
// Version 2.0
//////////////////////////////////////////////////////////////////

#define _CRTDBMAP_ALLOC // For debug purposes
#include "LSTMNet.h"
#include "GRUNet.h"
#include "CumulusIA.h"
#include <fstream>
#include <stdlib.h>

#include <vector>
#include <sstream>

// JMCT para espera en video
#include <thread>

#include <chrono>
#include <random>

#include "Defines.h"
#include "Main.h"
#include "Camera.h"
#include "FluidCPU.h"
#include "FluidCUDA.h"
#ifdef CUDA
#include "PrecomputeCUDA.h"
#endif
#ifdef CPU
#include "PrecomputeCPU.h"
#endif 

#ifdef __GNUC__
#include <time.h>
#include<unistd.h>
#include <chrono>
#endif

std::ofstream outdata;

struct AtExit // For debug purposes
{
// JMCT
#ifdef _MSC_VER
	~AtExit() { _CrtDumpMemoryLeaks(); }
#endif
} doAtExit;

// Sun/moon light direction
glm::vec3 sunDir = glm::normalize(glm::vec3(0.5, -0.2, 1.0));

#ifdef FLUID
const float windForce = 0.1f;
const int FLUIDLIMIT = 100;
#endif
bool parallel = true;
bool mean = false;

nimbus::Shader shaderCloud;
nimbus::Shader shaderAxis;
nimbus::Shader shaderSky;
nimbus::Canvas canvas;
#ifdef MOUNT
nimbus::Mountain mountain;
#endif
nimbus::Axis axis;
nimbus::Model model1;
nimbus::Model model2;
nimbus::Morph morphing;

#ifdef NEURAL
nimbus::CumulusIA myCloud[10];
#else
nimbus::Cumulus myCloud[10]; // Create cumulus clouds
#endif
glm::vec3 EXTINCTION_LIMIT(100.0, 15.0, 30.0); // Limit of scenary
const GLint SCR_W = 1200;  // Screen dimensions
const GLint SCR_H = 600;
const GLfloat SCR_Z = 480.0;  // Depth of camera //AXIS_Z-SCR_Z 890.0
const int TEXTSIZ = 128; // Size of cloud base texture for a later fBm
bool EVOLUTE = true; // If morphing evolution

float alpha; // Linear interpolation increment for morphing
float alphaDir; // Linear interpolation direction for morphing

// Wind direction
#ifdef FLUID
nimbus::Winds windDirection = nimbus::Winds::WEST;
#endif
#ifdef NEURAL
nimbus::Winds windDirection = nimbus::Winds::WEST;
#endif
// 30 FPS minimum real-time
// JMCT
#ifdef _WIN32
const DWORD FPS = 1000 / 30
#elif __linux_
const unsigned int FPS = 1000 / 30
#endif
;

int windowWidth =  SCR_W;
int windowHeight = SCR_H;
int windowHandle = 0;

// Required cameras

nimbus::Camera cameraFrame;
nimbus::Camera cameraSky;
nimbus::Camera cameraAxis;

// Main camera positions
glm::vec3 userCameraPos;

// Camera initial position
glm::vec3 initialCameraFramePosition;
glm::vec2 mousePos;


//////////////////////////////////////// FLUIDS ////////////////////////////////////////////
#ifdef FLUID
float dt = 0.4f; // time delta
float diff = 0.0f; // diffuse
float visc = 0.00001f; // viscosity

#ifdef CUDA
nimbus::FluidCUDA windGridCUDA(dt, diff, visc);
#endif
#ifdef CPU
nimbus::FluidCPU windGridCPU(dt, diff, visc);
#endif
#endif

#ifdef CUDA
nimbus::PrecomputeCUDA precompCUDA;
#endif
#ifdef CPU
nimbus::PrecomputeCPU precompCPU;
#endif

float cloudDepth = 20.0f; // Distance from viewer
float frameCount = 0.0; //FPS measurement
int previousTime;

int  skyTurn = 0; // Day hour
float timeDelta = 99999.0f; // Timers 
float timeDir = -0.06f;
int simcont = 1000;

const int TOTALTIME = 5;
const int PRECOMPTIMEOUT = 150;

int totalTime = 0; // Time to check born/extinction
int precomputeTimeOut = PRECOMPTIMEOUT; // Time for regular precompute light

bool debug = false; // If debugging
bool onPlay = true; // The loop is idle
bool firstPass = true; // First pass in loop

// Function prototypes

void reshapeGL(int w, int h);
void displayGL();
void idleGL();
void keyboardGL(unsigned char c, int x, int y);
void keyboardUpGL(unsigned char c, int x, int y);
void specialGL(int key, int x, int y);
void specialUpGL(int key, int x, int y);
void mouseWheel(int button, int dir, int x, int y);
void motionGL(int x, int y);
void applyWind();
void syncFPS();

#ifdef NEURAL

NeuralNet *net;

int NUM_SPH = 60;
std::string net_type = "LSTM";
int hidden_dim;
int n_layers; 
std::string input_file = "";

enum ParamsFromFilename {NO_PT, PT, DEFAULT};

// bool realistic = false;
int call_infer = 1;
int display_step = 0;

const int netM = 30;
const int netN = 7;
const int netO = 7;
float dt = 0.4f; // time delta
float diff = 0.0f; // diffuse
float visc = 0.00001f; // viscosity
float force = 0.2f;
float cd = 0.03f;

//JMCT in reshape_func
bool start = true;
float sleep_secs = 0.1;

#ifdef CUDA
nimbus::FluidCUDA windGridCUDA(dt, diff, visc);
#endif
#ifdef CPU
nimbus::FluidCPU windGridCPU(dt, diff, visc);
#endif


// Ask torch caracteristics and
// Set torch device type (CPU, CUDA) for neural network
void initTorch()
{
	std::cout << "PyTorch version: "
      << TORCH_VERSION_MAJOR << "."
      << TORCH_VERSION_MINOR << "."
      << TORCH_VERSION_PATCH
      << " CUDA_VERSION " << CUDA_VERSION << std::endl;
    std::string device_name = "cpu"; 
    NeuralNet::setDeviceName(device_name);
	try {
		std::cout << torch::cuda::is_available() << std::endl;
        torch::Device device = torch::Device(device_name);
        torch::Tensor tensor = at::tensor({ -1, 1 });
        tensor.to(device);
	}
	catch (std::exception& ex) {
		std::cout << ex.what() << std::endl;
	}

	torch::Tensor tensor = torch::eye(3);
	std::cout << tensor << std::endl;
	std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;

	if (torch::cuda::is_available()) {
		std::cout << "cudnn_is_available" << std::endl;
	}

}

//Creates neural network from paramenters read from .pt file name

void initNet(ParamsFromFilename params_from_filename, std::string input_file) {
    dt = 0.4f; // time delta
    diff = 0.0f; // diffuse
    visc = 0.00001f; // viscosity
    force = -0.2f;
    cd =  0.1f;
    
   if(params_from_filename == PT) {
        NeuralNet::setNUM_SPH(NUM_SPH);
        if(net_type.compare("LSTM") == 0)
            net = new LSTMNet(3*NUM_SPH, 3*NUM_SPH, hidden_dim, n_layers, force);
        else if(net_type.compare("GRU") == 0)
            net = new GRUNet(3*NUM_SPH, 3*NUM_SPH, hidden_dim, n_layers, force);
        net->loadWeigths(input_file);
        std::cout << "input_file " << input_file << std::endl;
        std::cout << "NUM_SPH " << NUM_SPH << std::endl;
        std::cout << "n_layers " << n_layers << std::endl;
        std::cout << "hidden_dim " << hidden_dim << std::endl;
    }     
    
    try {
        net->loadWeigths(input_file);
    }
    catch (...) {
        std::cout << std::endl << "Net file " << input_file << " not found" << std::endl;
        return;
    }
    
    std::cout << net->parameters().size() << std::endl;
    
    if(!net->setFluidData(netM, netN, netO, dt, diff, visc)) {
        std::cout << "Cannot allocateGridData for neural network" << std:: endl;
        return;
    }
    net->initWindGrid();
    
}
#endif

// Initialize the OpenGL context and create a render window.

void initGL(int argc, char* argv[])
{
	std::cout << "Initialize OpenGL..." << std::endl;

	glutInit(&argc, argv);

	int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
	int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);

	mousePos = glm::vec2(SCR_W / 2.0, SCR_H / 2.0);


	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);

	// Create an OpenGL 4.5 core forward compatible context.
	glutInitContextVersion(4, 5);
	glutInitContextProfile(GLUT_CORE_PROFILE);
	glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);

	glutInitWindowPosition((screenWidth - windowWidth) / 2, (screenHeight - windowHeight) / 2);
	glutInitWindowSize(windowWidth, windowHeight);

	std::string title = "Real-time ontogenetic cloud modelling -- UNED Thesis. Rendering at " + std::to_string(SCR_W) + " X " + std::to_string(SCR_H) + ".";
	windowHandle = glutCreateWindow(title.c_str());

	// Register GLUT callbacks.
	glutIdleFunc(idleGL);
	glutDisplayFunc(displayGL);
	glutKeyboardFunc(keyboardGL);
	glutKeyboardUpFunc(keyboardUpGL);
	glutSpecialFunc(specialGL);
	glutSpecialUpFunc(specialUpGL);
	glutMouseWheelFunc(mouseWheel);
	glutMotionFunc(motionGL);
	glutReshapeFunc(reshapeGL);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
	glClearColor(0.46f, 0.78f, 0.97f, 0.0f);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_TEXTURE_3D);
	glFrontFace(GL_CW);
	std::cout << "Initialize OpenGL Success!" << std::endl;
}


// Initialize Glew

void initGLEW()
{

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK)
	{
		std::cerr << "There was a problem initializing GLEW. Exiting..." << std::endl;
		exit(-1);
	}

	// Check for 3.3 support.
	// I've specified that a 3.3 forward-compatible context should be created.
	// so this parameter check should always pass if our context creation passed.
	// If we need access to deprecated features of OpenGL, we should check
	// the state of the GL_ARB_compatibility extension.
	if (!GLEW_VERSION_4_5)
	{
		std::cerr << "OpenGL 4.5 required version support not present." << std::endl;
		exit(-1);
	}
// JMCT
#ifdef _WIN32
	if (WGLEW_EXT_swap_control)
	{
		wglSwapIntervalEXT(0); // Disable vertical sync
	}
#elif __linux_
	if (GLX_EXT_swap_control)
	{
        Display *dpy = glXGetCurrentDisplay();
        GLXDrawable drawable = glXGetCurrentDrawable();
        if (drawable)
            glXSwapIntervalEXT(dpy, drawable, 0); // Disable vertical sync
        else
            std::cout << "glXSwapIntervalEXT() failed" << std::endl;
	}
#endif

	const GLubyte *ext;
	ext = glGetStringi(GL_EXTENSIONS, 0);

}

#ifdef NEURAL

// Set the neural network for every cloud

void initialiceCloudNet(int index) {
    myCloud[index].setNetGridValues(netM, netN, netO, force, cd/*, realistic*/);
    myCloud[index].setNeuralNet(net);

    myCloud[index].initialState();
    myCloud[index].calculateMeans();    
}

inline char separator()
{
#ifdef _WIN32
    return '/';
#else
    return '/';
#endif
}

void split_str( std::string const &str, const char delim, std::vector <std::string> &out )  
{  
    // create a stream from the string  
    std::stringstream s(str);  
        
    std::string s2;  
    while (std:: getline (s, s2, delim) )  
    {  
        out.push_back(s2); // store the string in s2  
    }  
} 

// Read net parameters from .pt file name separated by _, net_type, NUM_SPH, n_layers, hidden_dim

ParamsFromFilename get_params_from_filename()
{
    std::vector <std::string> out;
    const char delim = '_';
    if(input_file.empty()) {
        std::cout << "Path to .pt file" << std::endl;    
        getline(std::cin, input_file);
        if(input_file.empty())
            return DEFAULT;
    }
    std::string name = input_file.substr(input_file.find_last_of(separator()) + 1);
    std::string extension = name.substr(name.find_last_of(".") + 1);
    
    if(extension != "pt") {
        std::cout << "Path is not from a .pt file" << std::endl;
        return NO_PT;
    }
    name = name.substr(0, name.find_last_of("."));    
    split_str(name, delim, out);

    int i = 0;
    for (const auto &s: out) {  
        try {
            if(i == 0) {
                if(s == "lstm")
                    net_type = "LSTM";
                else if(s == "gru")
                    net_type = "GRU";
                else {
                    std::cout << "Invalid net type" << std::endl;
                    return NO_PT;
                }
            }
            else if(i == 1) {
                NUM_SPH = stoi(s);
            }
            else if(i == 2) {
                n_layers = stoi(s);
            }
            else if(i == 3) {
                hidden_dim = stoi(s);
            }            
        }
        catch (std::exception& e) {
            std::cout << e.what() << '\n';
        }
        ++i;
    }
    
    return PT;
}

#endif

// Entry point for cumulus rendering
#ifdef CUMULUS

int main(int argc, char* argv[])
{
	// Initialize FreeGLUT and Glew
#ifdef NEURAL
    // Set the path to .pt file
//     input_file = "../Nube/x64/data/neural/lstm_35_5_350.pt";
    input_file = "/home/jose/carlos_jimenez_nubes/modelos/lstm_35_5_350.pt";
	
	ParamsFromFilename params_from_filename = get_params_from_filename();
    if(params_from_filename == NO_PT) {
        return 1;
    }
      
	initTorch();
#endif
	initGL(argc, argv);
	initGLEW();
//     outdata.open("datos.txt"); // opens the file

	try {
		// Cameras setup
		cameraFrame.setProjectionRH(30.0f, SCR_W / SCR_H, 0.1f, 2000.0f);
		cameraAxis.setProjectionRH(30.0f, SCR_W / SCR_H, 0.1f, 2000.0f);
		cameraFrame.setViewport(0, 0, SCR_W, SCR_H);
		cameraFrame.setLookAt(glm::vec3(0, 0, -SCR_Z), glm::vec3(0, 0, SCR_Z));
		cameraFrame.translate(glm::vec3(-SCR_W / 2.0,-SCR_H / 2.0, -SCR_Z));
	
		userCameraPos = glm::vec3(0.0, 0.4, 0.0);

		// Create framgent shader canvas
		canvas.create(SCR_W, SCR_H);
		// Create cloud base texture
		nimbus::Cloud::createTexture(TEXTSIZ);

#ifdef MOUNT
		mountain.create(800.0, false); // Create mountain
#endif
		axis.create(); // Create 3D axis

		//Create cumulus clouds
        glm::vec3 center0 = glm::vec3(-10,5,0);
		myCloud[0].create(35, 2.8, center0, 0.0, 4.2, 0.0, 1.9, 0.0, 3.2, true, false);

		// Calculate guide points for cumulus

		myCloud[0].setGuidePoint(nimbus::Winds::WEST);
	
#ifdef NEURAL
        initNet(params_from_filename, input_file);
        for(int i = 0; i < nimbus::Cloud::getNumClouds(); i++) 
            initialiceCloudNet(i);
#endif        
		// Load shaders
		// Main shader
		shaderCloud.loadShader(GL_VERTEX_SHADER, "../Nube/x64/data/shaders/canvasCloud.vert");
#ifdef MOUNT
		// Mountains shader for cumulus
		shaderCloud.loadShader(GL_FRAGMENT_SHADER, "../Nube/x64/data/shaders/clouds_CUMULUS_MOUNT.frag");
#endif
#ifdef SEA
		// Sea shader for cumulus
		shaderCloud.loadShader(GL_FRAGMENT_SHADER, "../Nube/x64/data/shaders/clouds_CUMULUS_SEA.frag");
#endif
		// Axis shaders
		shaderAxis.loadShader(GL_VERTEX_SHADER, "../Nube/x64/data/shaders/axis.vert");
		shaderAxis.loadShader(GL_FRAGMENT_SHADER, "../Nube/x64/data/shaders/axis.frag");

		// Create shader programs
		shaderCloud.createShaderProgram();
		shaderAxis.createShaderProgram();

#ifdef MOUNT
		mountain.getUniforms(shaderCloud);
#endif
		canvas.getUniforms(shaderCloud);
		nimbus::Cloud::getUniforms(shaderCloud);
		nimbus::Cumulus::getUniforms(shaderCloud);
		axis.getUniforms(shaderAxis);
        
		// Start main loop
		glutMainLoop();

	}
	catch (nimbus::NimbusException& exception)
	{
		exception.printError();
		system("pause");
	}

	// Free texture
	nimbus::Cloud::freeTexture();
    
    //JMCT 
    return 0;

}

#endif


// Entry point for mesh morphing
#ifdef MODEL

void main(int argc, char* argv[])
{
	// Initialize FreeGLUT and Glew

	initGL(argc, argv);
	initGLEW();

	try {
		// Cameras setup
		cameraFrame.setProjectionRH(30.0f, SCR_W / SCR_H, 0.1f, 2000.0f);
		cameraAxis.setProjectionRH(30.0f, SCR_W / SCR_H, 0.1f, 2000.0f);
		cameraFrame.setViewport(0, 0, SCR_W, SCR_H);
		cameraFrame.setLookAt(glm::vec3(0, 0, -SCR_Z), glm::vec3(0, 0, SCR_Z));
		cameraFrame.translate(glm::vec3(-SCR_W / 2.0, -SCR_H / 2.0, -SCR_Z));
		
		userCameraPos = glm::vec3(0.0, 0.4, 0.0);

		// Create framgent shader canvas
		canvas.create(SCR_W, SCR_H);
		// Create cloud base texture
		nimbus::Cloud::createTexture(TEXTSIZ);

#ifdef MOUNT
		mountain.create(300.0, false); // Create mountain
#endif
		axis.create(); // Create 3D axis
		model1.create(glm::vec3(-1.0, 7.0, 0.0), MESH1, 1.1f); // Create mesh 1
		model2.create(glm::vec3(1.0, 7.0, -3.0), MESH2, 1.1f);  // Create mesh 2
		morphing.setModels(&model1, &model2, EVOLUTE); // Setup modes for morphing

		// Load shaders
		// Main shader
		shaderCloud.loadShader(GL_VERTEX_SHADER, "../Nube/x64/data/shaders/canvasCloud.vert");
#ifdef MOUNT
		// Mountains shader for 3D meshes based clouds
		shaderCloud.loadShader(GL_FRAGMENT_SHADER, "../Nube/x64/data/shaders/clouds_MORPH_MOUNT.frag");
#endif
#ifdef SEA
		// Sea shader for 3D meshes based clouds
		shaderCloud.loadShader(GL_FRAGMENT_SHADER, "../Nube/x64/data/shaders/clouds_MORPH_SEA.frag");
#endif

		// Axis shaders
		shaderAxis.loadShader(GL_VERTEX_SHADER, "../Nube/x64/data/shaders/axis.vert");
		shaderAxis.loadShader(GL_FRAGMENT_SHADER, "../Nube/x64/data/shaders/axis.frag");

		// Create shader programs
		shaderCloud.createShaderProgram();
		shaderAxis.createShaderProgram();

		// Locate uniforms
		nimbus::Cloud::getUniforms(shaderCloud);

#ifdef MOUNT

		mountain.getUniforms(shaderCloud);

#endif

		canvas.getUniforms(shaderCloud);

		nimbus::Model::getUniforms(shaderCloud);

		axis.getUniforms(shaderAxis);

		// Start main loop

		glutMainLoop();

	}
	catch (nimbus::NimbusException& exception)
	{
		exception.printError();
		system("pause");
	}

	// Free texture
	nimbus::Cloud::freeTexture();
}

#endif


// Reshape window

void reshapeGL(int w, int h)
{
	if (h == 0)
	{
		h = 1;
	}

	windowWidth = w;
	windowHeight = h;

	glutPostRedisplay();
}

#ifdef __linux_
unsigned int GetTickCountMs()
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts))
        return 0;

    return (unsigned int)(ts.tv_nsec / 1000000) + ((unsigned int)ts.tv_sec * 1000ull);
}
#endif

// Application speed synchronization

void syncFPS()
{
#ifdef _WIN32
	static DWORD dwLastTime = 0;

	DWORD dwCurrentTime = GetTickCount64(); // Get milliseconds from the system start up

	DWORD dwElapsed = dwCurrentTime - dwLastTime; // Calculates elapsed time

	if (dwElapsed < FPS) // The frame loop lasted less than the defined time 				
	{
		Sleep(FPS - dwElapsed);	// Sleeps the application
		dwLastTime = dwCurrentTime + FPS - dwElapsed; // Adds the sleeped time

	}
	else dwLastTime = dwCurrentTime;	// The frame loop exceeded the time
#elif __linux_
	static unsigned int dwLastTime = 0;
    unsigned int dwCurrentTime = GetTickCountMs();
    unsigned int dwElapsed = dwCurrentTime - dwLastTime;
    if (dwElapsed < FPS) // The frame loop lasted less than the defined time 				
	{
		usleep((FPS - dwElapsed) * 1000);	// Sleeps the application
		dwLastTime = dwCurrentTime + FPS - dwElapsed; // Adds the sleeped time
	}
	else dwLastTime = dwCurrentTime;
#endif
}

// Calculate Frame-per-second
float fps;

void calculateFPS()
{
	static bool firstTime = true;

	int currentTime;

	//  Increase frame count
	frameCount++;

	//  Get the number of milliseconds since glutInit called
	//  (or first call to glutGet(GLUT ELAPSED TIME)).
	currentTime = glutGet(GLUT_ELAPSED_TIME);

	if (firstTime)
	{
		previousTime = currentTime;
		firstTime = false;
	}

	//  Calculate time passed
	int timeInterval = currentTime - previousTime;

	if (timeInterval > 1000)
	{
		//  calculate the number of frames per second
		fps = frameCount / (timeInterval / 1000.0f);
		std::cout << "FPS = " << fps << std::endl;

		//  Set time
		previousTime = currentTime;

		//  Reset frame count
		frameCount = 0;

	}
}

#ifdef CUMULUS 
#ifdef FLUID
void applyWind() // Apply wind force
{

	if (parallel)
	{
		windGridCUDA.clearUVW(); // Clear fluid internal data
		for (int i = 0; i < nimbus::Cumulus::getNumClouds(); i++)
			myCloud[i].applyWind(windForce, &windGridCUDA);
	} else
	{
		windGridCPU.clearUVW(); // Clear fluid internal data
		for (int i = 0; i < nimbus::Cumulus::getNumClouds(); i++)
			myCloud[i].applyWind(windForce, &windGridCPU);
	}
	
}

#endif
#endif

#ifdef CUMULUS

// Render function
void displayGL()
{
//      std::this_thread::sleep_for(std::chrono::milliseconds(100));
	if (onPlay)
	{

		// Clear back-buffer
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//////////////// CLOUDS ///////////////////

		// Change coordinate system

		glm::vec2 mouseScale = mousePos / glm::vec2(SCR_W, SCR_H);

		// Rotate camera
		glm::vec3 userCameraRotatePos = /*userCameraPos + glm::vec3(0, 0, 2);*/ glm::vec3(sin(mouseScale.x*3.0), mouseScale.y, cos(mouseScale.x*3.0));
		
		glDisable(GL_DEPTH_TEST);

		shaderCloud.useProgram();

		if (firstPass) // If first iteration
		{

			// Render cloud base texture
			nimbus::Cloud::renderTexture();

#ifdef MOUNT
			// Render mountain
			mountain.render();
#else
			glActiveTexture(GL_TEXTURE0 + 1);
#endif
			nimbus::Cloud::renderFirstTime(SCR_W, SCR_H);
		}
		
			// Render cloud base class uniforms
			// JMCT
			glm::vec3 cameraRotatePos = cloudDepth * userCameraRotatePos;
			nimbus::Cloud::render(mousePos, timeDelta*2.0, cloudDepth, skyTurn, cameraRotatePos, debug);
		
#ifdef FLUID

			// Calculate and apply wind
			for (int i = 0; i < nimbus::Cumulus::getNumClouds(); i++)
				(parallel) ? myCloud[i].computeWind(&windGridCUDA) : myCloud[i].computeWind(&windGridCPU);
			
#endif

			// Wind setup
			nimbus::Cumulus::setWind(windDirection);            
          
			// Render cumulus class
			nimbus::Cumulus::render(shaderCloud);

			if (firstPass) //precomputeTimeOut >= PRECOMPTIMEOUT) // Check for regular precompute light (shading)
			{
	
				std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

				if (parallel)
				{
					if (skyTurn == 0) // If morning sun is near else sun is far (sunset)
						nimbus::Cumulus::precomputeLight(precompCUDA, sunDir, 100.0f, 1e-9f);
					else nimbus::Cumulus::precomputeLight(precompCUDA, sunDir, 10000.0f, 1e-6f);
					cudaDeviceSynchronize();
				}
				else
				{
					if (skyTurn == 0)
						nimbus::Cumulus::precomputeLight(precompCPU, sunDir, 100.0f, 0.4f);
					else nimbus::Cumulus::precomputeLight(precompCPU, sunDir, 10000.0f, 1e-6f);
				}
				std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				std::cout << "Time difference PRECOMPUTE LIGHT = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
				
				for (int i = 0; i < nimbus::Cumulus::getNumClouds(); i++)
					nimbus::Cloud::renderVoxelTexture(i);
			
				precomputeTimeOut = 0;
	
			}


			if (totalTime > TOTALTIME)
			{
				timeDelta += nimbus::Cumulus::getTimeDir();
				totalTime = 0;
			}			

			totalTime++;
			precomputeTimeOut++;

			// User camera setup
			cameraSky.setLookAt(cloudDepth * userCameraRotatePos, userCameraPos);
			cameraSky.translate(userCameraPos);
				
			canvas.render(cameraFrame, cameraSky);

			/////////////// AXIS ////////////////////

			shaderAxis.useProgram();

			// Render axis
			cameraAxis.setViewport(SCR_W / 3, SCR_H / 3, SCR_W, SCR_H);
			cameraAxis.setLookAt(10.0f * userCameraRotatePos, userCameraPos);
			cameraAxis.setPosition(userCameraPos);

			axis.render(cameraAxis);

			// Restore landscape viewport
			cameraFrame.setViewport(0, 0, SCR_W, SCR_H);
			glutSwapBuffers();

			glEnable(GL_DEPTH_TEST);
			//  Calculate FPS
			calculateFPS();
			firstPass = false; // First pass ended
		
	}
}

#endif

#ifdef MODEL

void displayGL()
{
	if (onPlay)
	{

		// Clear back-buffer
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Change coordinate system

		glm::vec2 mouseScale = mousePos / glm::vec2(SCR_W, SCR_H);

		// Rotate camera

		glm::vec3 userCameraRotatePos =  glm::vec3(sin(mouseScale.x*3.0), mouseScale.y, cos(mouseScale.x*3.0));
		
		//////////////// MORPHING ///////////////////

		glDisable(GL_DEPTH_TEST);

		shaderCloud.useProgram();
		if (firstPass) // If first iteration
		{

			nimbus::Cloud::renderFirstTime(SCR_W, SCR_H);

			clock_t start = clock();
#ifdef CUDA	
			// Precompute light for meshes
			nimbus::Model::precomputeLight(precompCUDA, sunDir, 100.0f, 1e-6f, model1.getNumEllipsoids(), model2.getNumEllipsoids());
#else
			nimbus::Model::precomputeLight(precompCPU, sunDir, 100.0f, 1e-6f, model1.getNumEllipsoids(), model2.getNumEllipsoids());
#endif
			clock_t end = clock();
			std::cout << "PRECOMPUTE TIME LIGHT = " << end - start << std::endl;

			// Prepare for morphing
			(EVOLUTE) ? morphing.prepareMorphEvolute() : morphing.prepareMorphInvolute();
			alpha = alphaDir = 0.01f;
			// First morphing render
			nimbus::Model::renderFirstTime(model2.getNumEllipsoids(), EVOLUTE);
			
			// Render cloud base texture
			nimbus::Cloud::renderTexture();
#ifdef MOUNT
			// Render mountain
			mountain.render();
#endif
		
			// Render clouds precomputed light textures
			for (int i = 0; i < nimbus::Cloud::getNumClouds(); i++)
				nimbus::Cloud::renderVoxelTexture(i);
		}

		// Render cloud base class uniforms
		nimbus::Cloud::render(mousePos, timeDelta, cloudDepth, skyTurn, cloudDepth * userCameraRotatePos, debug);
			
		static bool totalTimePass = false;

		if (totalTime > TOTALTIME) // Check time for morphing animation
		{
			totalTimePass = true;
			timeDelta += timeDir;
			if (alpha < 1.0 && alpha > 0.0)
			{
				alpha += alphaDir; // Animate morphing
				(EVOLUTE) ? morphing.morphEvolute(alpha) : morphing.morphInvolute(alpha);
				morphing.morph(0.1f); // Animation speed
			}

			totalTime = 0;
		}

		totalTime++;

		// Mesh renderer
		nimbus::Model::render(shaderCloud, (totalTimePass) ? morphing.getCloudPosRDst() : nimbus::Model::getCloudPosR(), morphing.getCloudPosDst(), alpha);


		// User camera setup
		cameraSky.setLookAt(cloudDepth * userCameraRotatePos, userCameraPos);
		cameraSky.translate(userCameraPos);

		canvas.render(cameraFrame, cameraSky);

		/////////////// AXIS ////////////////////

		shaderAxis.useProgram();

		// Render axis
		cameraAxis.setViewport(SCR_W / 3, SCR_H / 3, SCR_W, SCR_H);
		cameraAxis.setLookAt(10.0f * userCameraRotatePos, userCameraPos);
		cameraAxis.setPosition(userCameraPos);

		axis.render(cameraAxis);

		// Restore landscape viewport
		cameraFrame.setViewport(0, 0, SCR_W, SCR_H);

		glutSwapBuffers();

		glEnable(GL_DEPTH_TEST);
		//  Calculate FPS
		calculateFPS();
		firstPass = false; // First pass ended
	}
}

#endif


// Idle function

void idleGL()
{
	if (!onPlay) return;

	syncFPS();

#ifdef CUMULUS
#ifdef FLUID

	simcont++;
	if (simcont > FLUIDLIMIT) // Simulate fluid
	{
		applyWind();
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		if (parallel)
		{
			windGridCUDA.sendData();
			windGridCUDA.sim();
			windGridCUDA.receiveData();
			cudaDeviceSynchronize(); // For clock_t usage
		} else windGridCPU.sim();
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		float ellapsed = std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();
		
		static float samples = 0.0f;
		static float sum = 0.0f;

		if (mean)
		{
			sum += ellapsed;
			samples++;
			if (samples > 10)
			{
				std::cout << "====================MEAN for" << ((parallel) ? " CUDA = " : " CPU = ") << sum / samples << std::endl;
				sum = samples = 0.0f;
				mean = false;
			}
		}
		else
		{
			samples = 0.0f;
			sum = 0.0f;
		}


		simcont = 0;
	}
#endif
#ifdef NEURAL
        int num_clouds = nimbus::Cumulus::getNumClouds();
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		for(int i = 0; i < num_clouds; i++) { 
            myCloud[i].inferPos();
         };
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		float ellapsed = std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();


		static float samples = 0.0f;
		static float sum = 0.0f;

		if (mean)
		{
			sum += ellapsed;
			samples++;
			if (samples > 30)
			{
				//std::cout << "====================MEAN FPS for" << ((parallel) ? " CUDA = " : " CPU = ") << sumFPS / samples << std::endl;
				std::cout << "Time difference MEAN NEURAL TIME = " << sum / samples << "[ms]" << "\t" << std::endl;
				sum = samples = 0.0f;
				mean = false;
			}
		}
		else
		{
			samples = 0.0f;
			sum = 0.0f;
		}
		

		for (int i = 0; i < num_clouds; i++) {
			myCloud[i].updatePosition();
		};

#endif  
#endif
	glutPostRedisplay();
}

// Key press function



void keyboardGL(unsigned char c, int x, int y)
{
	switch (c)
	{
	case 'w':
	case 'W':
		break;
	case 'a':
	case 'A':
		break;
	case 's':
		break;
	case 'S':
		break;
	case 't':
		parallel = !parallel;
		(parallel) ? std::cout << "......CUDA MODE......." << std::endl : std::cout << "......CPU MODE......." << std::endl;
		break;
	case 'm':
		mean = !mean;
		(mean) ? std::cout << "MEAN ENABLED for " : std::cout << "MEAN DISABLED for ";
		std::cout << ((parallel) ? "CUDA" : "CPU") << std::endl;

		break;
	case 'd':
		break;
	case 'D':
		break;
	case 'q':
	case 'Q':
		break;
	case 'e':
	case 'E':
		break;
#ifdef CUMULUS
	case '8': // The wind blows from the south
		windDirection = nimbus::Winds::SOUTH;
		for (int i = 0; i < nimbus::Cloud::getNumClouds(); i++)
			myCloud[i].setGuidePoint(windDirection);
		break;
	case '2': // The wind blows from the north
		windDirection = nimbus::Winds::NORTH;
		for (int i = 0; i < nimbus::Cloud::getNumClouds(); i++)
			myCloud[i].setGuidePoint(windDirection);

		break;
	case '4': // The wind blows from the east
		windDirection = nimbus::Winds::EAST;
		for (int i = 0; i < nimbus::Cloud::getNumClouds(); i++)
			myCloud[i].setGuidePoint(windDirection);

		break;
	case '6': // the wind blows from the west
		windDirection = nimbus::Winds::WEST;
		for (int i = 0; i < nimbus::Cloud::getNumClouds(); i++)
			myCloud[i].setGuidePoint(windDirection);
		break;
	case 'z':
		break;
	case 'n':
		break;
#endif
	case 'r': {// Recreate new cloud
		nimbus::Cloud::resetNumClouds();
		precomputeTimeOut = PRECOMPTIMEOUT;
        glm::vec3 center0 = glm::vec3(5.0, 9.0, 0.0);
		myCloud[0].create(35, 2.7f, center0, 0.0f, 4.0f, 0.0f, 2.0f, 0.0f, 3.2f, true, false);
        glm::vec3 center1 = glm::vec3(-13.0, 15.0, 0);
		myCloud[1].create(35, 2.7f, center1, 0.0f, 4.0f, 0.0f, 2.0f, 0.0f, 3.2f, true, false);
        glm::vec3 center2 = glm::vec3(10.0, -5.0, 0.0);
		myCloud[2].create(35, 2.7f, center2, 0.0f, 4.0f, 0.0f, 2.0f, 0.0f, 3.0f, true, false);
        glm::vec3 center3 = glm::vec3(-20.0, 9.0, 0.0);
		myCloud[3].create(35, 2.7f, center3, 0.0f, 4.0f, 0.0f, 2.0f, 0.0f, 3.0f, true, false);   

#ifdef NEURAL
        for(int i = 0; i < nimbus::Cloud::getNumClouds(); i++) 
            initialiceCloudNet(i);
#endif

		windDirection = nimbus::Winds::WEST;
		for (int i = 0; i < nimbus::Cloud::getNumClouds(); i++)
			myCloud[i].setGuidePoint(windDirection);

		firstPass = true;
		break;
    }

	case 27:
		glutLeaveMainLoop();
		break;
	case 32:
		break;
	case 13: // Change daytime
		skyTurn = (skyTurn + 1) % 3;
	}
}

// Change linear interpolation direction back/front<->front/back

void changeADir()
{
	if (alpha <= 0.0)
	{
		alpha = 0.01f;
		alphaDir = 0.01f;
	}
	else if (alpha >= 1.0)
	{
		alpha = 0.99f;
		alphaDir = -0.01f;
	}
}

// Keyboard press

void keyboardUpGL(unsigned char c, int x, int y)
{
	switch (c)
	{
	case 'w':
		break;
	case 'W':
		break;
	case 'a':
	case 'A':
		break;
	case 's':
		break;
	case 'S':
		break;
	case 'd':
		debug = !debug; // Activating debug
		std::cout << "DEBUG = " << debug << std::endl;
		break;
	case 'D':
		break;
	case 'q':
		changeADir(); // Setup evolution/involution direction
		break;
	case 'Q':
		break;
	case 'e':
		alphaDir = -alphaDir; // Change evolution/involution direction
		break;
	case 'E':
		break;
	case 'p': // Pause 
		onPlay = !onPlay;
		break;
	default:
		break;
	}
}


// Special keys press
// Main camera movement

void specialGL(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_UP:
		userCameraPos.y += 0.1f;
		break;
	case GLUT_KEY_DOWN:
		userCameraPos.y -= 0.1f;
		break;
	case GLUT_KEY_RIGHT:
		userCameraPos.x += 0.1f;
		break;
	case GLUT_KEY_LEFT:
		userCameraPos.x -= 0.1f;

	}
}

void specialUpGL(int key, int x, int y)
{
}


// GLUT mouse wheel handler

void mouseWheel(int button, int dir, int x, int y)
{

	if (dir > 0)
	{
		// Camera zoom in
		cloudDepth -= 0.4f;
	}
	else
	{
		// Camera zoom out
		cloudDepth += 0.4f;
	}
	std::cout << "DEPTH = " << cloudDepth << std::endl;
}

// Mouse motion function

void motionGL(int x, int y)
{
	mousePos = glm::vec2(x * 4, y * 4);
}
