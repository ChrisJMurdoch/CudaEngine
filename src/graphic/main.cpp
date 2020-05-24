

// === INCLUDES ===

// OpenGL initialisation
#include <glad/glad.h>

// Window creation
#include <GLFW/glfw3.h>

// Math
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Engine
#include "..\..\include\graphic\main.hpp"
#include "..\..\include\graphic\util.hpp"
#include "..\..\include\generation\meshgen.hpp"
#include "..\..\include\generation\terrain.hpp"
#include "..\..\include\logger\log.hpp"
#include "..\..\include\models\vModel.hpp"
#include "..\..\include\models\eModel.hpp"


// === CONSTANTS ===

// Display constants
const int VIEW_WIDTH = 800;
const int VIEW_HEIGHT = 600;

// Vector constants
const glm::vec3 WORLD_UP = glm::vec3(0.0f, 1.0f,  0.0f);


// === VARIABLES ===

// Mouse
float yaw = -90, pitch = 0;

// Camera
glm::vec3 cameraPosition = glm::vec3(0.0f, 10.0f,  0.0f);
glm::vec3 cameraDirection = glm::vec3(0.0f, 0.0f, 1.0f);


// === FUNCTIONS ===

int main()
{
	// Initialise graphics
	GLFWwindow *window = NULL;
	GLuint programID;
	if ( initialise(window, programID) != 0 )
	{
		Log::print(Log::error, "Initialisation error.");
		glfwTerminate();
		return -1;
	}

	// Terrain data
	const int width = 180;
	const float tMin = 0, tMax = 20, tPeriod = 100;
	const float wMin = 2, wMax =  2.3, wPeriod =  25;
	int nVertices = pow(width-1, 2) * 6;

	// Terrain mesh
	float *terrainMap = new float[nVertices];
	terrain::generateHeightMap(width, tMin, tMax, terrainMap, terrain::sinXY, tPeriod);
	float *terrainMesh = new float[nVertices*6];
	meshgen::generateVertices(terrainMap, width, terrainMesh, meshgen::landscape);
	delete terrainMap;

	// Water mesh
	float *waterMap = new float[nVertices];
	terrain::generateHeightMap(width, wMin, wMax, waterMap, terrain::hashXY, wPeriod);
	float *waterMesh = new float[nVertices*6];
	meshgen::generateVertices(waterMap, width, waterMesh, meshgen::water);
	delete waterMap;

	// Create models
	VModel terrain = VModel( nVertices, terrainMesh, GL_STATIC_DRAW );
	delete terrainMesh;
	VModel water = VModel( nVertices, waterMesh, GL_STREAM_DRAW );
	delete waterMesh;

	// Model array
	const int nModels = 2;
	Model *models[nModels];
	models[0] = &terrain;
	models[1] = &water;

	// Main loop
	float lastTime = glfwGetTime();
	while( !glfwWindowShouldClose(window) )
	{
		// PHYSICS

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

		// RENDERING

		// Clear screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Use program shaders
		glUseProgram(programID);

		// COMMON TRANSFORMATIONS

		// View
		glm::mat4 view = glm::lookAt(cameraPosition, cameraPosition + cameraDirection, WORLD_UP);
		int modelLoc = glGetUniformLocation(programID, "view");
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(view));

		// Projection
		glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
		modelLoc = glGetUniformLocation(programID, "projection");
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(projection));

		// MODELS
		for(int i=0; i<nModels; i++)
		{
			// Model position
			glm::mat4 position = glm::mat4(1.0f);
			position = glm::scale(position, glm::vec3(0.1, 0.1, 0.1));
			modelLoc = glGetUniformLocation(programID, "model");
			glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(position));

			// Render
			models[i]->render();
		}

		// Unbind VAO
		glBindVertexArray(0);

		// Check inputs and display
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Cleanup
	glDeleteProgram(programID);
	glfwTerminate();

	return 0;
}

int initialise(GLFWwindow *&window, GLuint &programID)
{
	// Create window
	if ( createWindow(window) != 0 )
	{
		Log::print(Log::error, "Window creation error.");
		return -1;
	}

	// Initialise GLAD
	if ( initGLAD(window) != 0 )
	{
		Log::print(Log::error, "GLAD initialising error.");
		return -1;
	}

	// Load shaders
	programID = loadShaders( "shaders\\VShader.vert", "shaders\\FShader.frag" );
	if (programID == 0)
	{
		Log::print(Log::error, "Shader loading error.");
		return -1;
	}

	// Z-Buffering
	glEnable(GL_DEPTH_TEST);

	// Set background
	glClearColor(0.3f, 0.7f, 0.9f, 1.0f);

	return 0;
}

int createWindow(GLFWwindow *&window)
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
	window = glfwCreateWindow( VIEW_WIDTH, VIEW_HEIGHT, "CudaEngine", NULL, NULL);
	if (window == NULL)
		return -1;
	glfwMakeContextCurrent(window);

	// Setup key press mode
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(window, mouseCallback);
	
	return 0;
}

int initGLAD(GLFWwindow *window)
{
	// Load OpenGL
	if (gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) == 0)
		return -1;

	// Set viewport to full window and configure resize callback
	glViewport(0, 0, VIEW_WIDTH, VIEW_HEIGHT);
	glfwSetFramebufferSizeCallback(window, resizeCallback);

	return 0;
}

void resizeCallback(GLFWwindow *window, int width, int tMax)
{
    glViewport(0, 0, width, tMax);
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
	static float lastX = VIEW_WIDTH/2, lastY = VIEW_HEIGHT/2;
	static bool initialMouse = true;

	// Ignore mouse on first frame
	if (initialMouse)
	{
		lastX = xpos;
		lastY = ypos;
		initialMouse = false;
		return;
	}

	// Adjust camera angle
	const float sensitivity = 0.05f;
	yaw += (xpos - lastX) * sensitivity;
	pitch += (lastY - ypos) * sensitivity;
	lastX = xpos;
	lastY = ypos;

	// Constrain pitch
	if(pitch > 89.0f)
		pitch =  89.0f;
	if(pitch < -89.0f)
		pitch = -89.0f;
}

void processInput(GLFWwindow *window, float deltaTime, glm::vec3 cameraDirection)
{
	// Close window
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
	
	// Generate movement vectors
	const float cameraSpeed = 6.0f * deltaTime;
	const glm::vec3 forward = cameraSpeed * cameraDirection;
	const glm::vec3 right = cameraSpeed * glm::normalize(glm::cross(cameraDirection, WORLD_UP));
	const glm::vec3 up = cameraSpeed * glm::normalize(glm::cross(right, forward));

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
