
// === INCLUDES ===

// Header
#include "..\..\include\graphic\main.hpp"

// Engine
#include "..\..\include\generation\meshgen.hpp"
#include "..\..\include\graphic\io.hpp"
#include "..\..\include\graphic\util.hpp"
#include "..\..\include\logger\log.hpp"
#include "..\..\include\math\mathEngine.hpp"
#include "..\..\include\math\cpuMathEngine.hpp"
#include "..\..\include\math\gpuMathEngine.hpp"
#include "..\..\include\models\vModel.hpp"
#include "..\..\include\models\eModel.hpp"

// SIMD Vector math
#include <glm/gtc/matrix_transform.hpp>

// Standard headers
#include <string>
#include <thread>
#include <map>

// === CONSTANTS ===

// Vector constants
const glm::vec3 WORLD_UP = glm::vec3(0.0f, 1.0f,  0.0f);


// === VARIABLES ===

// Window size
int viewWidth = 800, viewHeight = 600;

// Mouse controls
float yaw = -90, pitch = 0;

// Camera
glm::vec3 cameraPosition = glm::vec3(0.0f, 8.0f,  0.0f);
glm::vec3 cameraDirection = glm::vec3(0.0f, 0.0f, 1.0f);


// === FUNCTIONS ===

int main( int argc, char *argv[] )
{
	// Determine math engine to use
	MathEngine *math;
	if ( argc>1 && strcmp( argv[1], "cuda" )==0 )
	{
		Log::print( Log::message, "Using Cuda acceleration" );
		math = &GPUMathEngine();
	}
	else
	{
		Log::print( Log::message, "Not using Cuda acceleration" );
		math = &CPUMathEngine();
	}

	// Start loading timer
	float startLoadTime = glfwGetTime();

	// Create window
	GLFWwindow *window = nullptr;
	createWindow(window);

	// Initialise GLAD
	initGLAD(window);

	// Enable Z-Buffering
	glEnable(GL_DEPTH_TEST);

	// Set background colour
	glClearColor(0.3f, 0.7f, 0.9f, 1.0f);

	// Load shader programs
	GLuint terrainProg, waterProg;
	loadShaders( "shaders\\Terrain.vert", "shaders\\FShader.frag", terrainProg );
	loadShaders( "shaders\\Water.vert", "shaders\\FShader.frag", waterProg );

	// Terrain data
	std::map<std::string, std::string> map = mapFile("assets/generation.kval");
	const int width = stoi(map["mapWidth"]);
	const float tMin = stof(map["terrainMin"]), tMax = stof(map["terrainMax"]), tPeriod = stof(map["terrainPeriod"]);
	const float wMin = stof(map["waterMin"]),   wMax = stof(map["waterMax"]),   wPeriod = stof(map["waterPeriod"]);
	int nVertices = pow(width-1, 2) * 6;

	// Generate heightmaps
	float *terrainMap = new float[nVertices], *waterMap = new float[nVertices];
	math->generateHeightMap(terrainMap, width, tMin, tMax, MathEngine::mountain, tPeriod, 6);
	math->generateHeightMap(waterMap, width, wMin, wMax, MathEngine::hash, wPeriod, 1);

	// Heightmaps => Meshes
	float *terrainMesh = new float[nVertices*6], *waterMesh = new float[nVertices*6];
	std::thread t1( meshgen::generateVertices, terrainMap, width, terrainMesh, meshgen::landscape );
	std::thread t2( meshgen::generateVertices, waterMap, width, waterMesh, meshgen::water );
	t1.join();
	t2.join();
	delete terrainMap;
	delete waterMap;

	// Meshes => Models
	VModel terrain = VModel( nVertices, terrainMesh, terrainProg, GL_STATIC_DRAW );
	VModel water = VModel( nVertices, waterMesh, waterProg, GL_STREAM_DRAW );
	delete terrainMesh;
	delete waterMesh;

	// Model array
	const int nModels = 2;
	Model *models[nModels];
	models[0] = &terrain;
	models[1] = &water;

	// End timer
	float endLoadTime = glfwGetTime();

	// Display load time
	Log::print( Log::message, "Loading time: ", Log::NO_NEWLINE);
	Log::print( Log::message, endLoadTime - startLoadTime, Log::NEWLINE);

	// Main loop
	float lastTime = glfwGetTime();
	while( !glfwWindowShouldClose(window) )
	{
		// Get time-delta
		float currentTime = glfwGetTime();
		float deltaTime = currentTime - lastTime;
		lastTime = currentTime;
		
		// Calculate new camera angle
		glm::vec3 direction;
		direction.x = cos( glm::radians(yaw)) * cos(glm::radians(pitch) );
		direction.y = sin( glm::radians(pitch) );
		direction.z = sin( glm::radians(yaw)) * cos(glm::radians(pitch) );
		glm::vec3 cameraDirection = glm::normalize(direction);

		// Move
		processInput(window, deltaTime, cameraDirection);

		// Clear screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Common matrices
		glm::mat4 view = glm::lookAt( cameraPosition, cameraPosition + cameraDirection, WORLD_UP );
		glm::mat4 projection = glm::perspective( glm::radians(60.0f), (float)viewWidth/(float)viewHeight, 0.1f, 1000.0f ); // Clip 10cm - 1km

		// MODELS
		for(int i=0; i<nModels; i++)
		{
			// Render model
			models[i]->render( currentTime, view, projection );
		}

		// Unbind VAO
		glBindVertexArray(0);

		// Check inputs and display
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Cleanup
	glDeleteProgram(terrainProg);
	glDeleteProgram(waterProg);
	glfwTerminate();

	return 0;
}

void createWindow(GLFWwindow *&window)
{
	// Initialise GLFW
	glfwInit();

	// Window settings
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	// Create window
	window = glfwCreateWindow( viewWidth, viewHeight, "CudaEngine", NULL, NULL);
	if ( window == nullptr )
		throw "Window creation error";
	glfwMakeContextCurrent(window);

	// Setup key press mode
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(window, mouseCallback);
}

void initGLAD(GLFWwindow *window)
{
	// Load OpenGL
	if ( gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) == 0 )
		throw "GLAD loading error";

	// Set viewport to full window and configure resize callback
	glViewport(0, 0, viewWidth, viewHeight);
	glfwSetFramebufferSizeCallback(window, resizeCallback);
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
	static float lastX = viewWidth/2, lastY = viewHeight/2;

	// Ignore mouse on first frame
	static bool initialMouse = true;
	if (initialMouse)
	{
		lastX = xpos;
		lastY = ypos;
		initialMouse = false;
		return;
	}

	// Adjust camera angle
	const float SENSITIVITY = 0.05f;
	yaw += (xpos - lastX) * SENSITIVITY;
	pitch += (lastY - ypos) * SENSITIVITY;
	lastX = xpos;
	lastY = ypos;

	// Constrain pitch
	pitch = pitch >  89.0f ?  89.0f : pitch;
	pitch = pitch < -89.0f ? -89.0f : pitch;
}

void resizeCallback(GLFWwindow *window, int width, int height)
{
	viewWidth=width, viewHeight=height;
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow *window, float deltaTime, glm::vec3 cameraDirection)
{
	// Close window
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
	
	// Generate movement vectors
	float cameraSpeed = 60.0f * deltaTime;
	glm::vec3 forward = cameraSpeed * cameraDirection;
	glm::vec3 right = cameraSpeed * glm::normalize(glm::cross(cameraDirection, WORLD_UP));
	glm::vec3 up = cameraSpeed * glm::normalize(glm::cross(right, forward));

	// Move
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPosition += forward;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPosition -= forward;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPosition -= right;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPosition += right;
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        cameraPosition += up;
	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        cameraPosition -= up;
}
