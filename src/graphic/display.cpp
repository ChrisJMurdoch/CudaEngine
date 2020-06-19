
// === INCLUDES ===

// Header
#include "..\..\include\graphic\display.hpp"

// Engine
#include "..\..\include\logger\log.hpp"

// SIMD Vector math
#include <glm/gtc/matrix_transform.hpp>

// Standard headers
#include <string>
#include <fstream>
#include <sstream>
#include <list>


// === CONSTANTS ===

// Vector constants
const glm::vec3 WORLD_UP = glm::vec3(0.0f, 1.0f,  0.0f);


// === VARIABLES ===

// Window
GLFWwindow *window = nullptr;
int viewWidth = 800, viewHeight = 600;

// Mouse controls
float yaw = -90, pitch = 0;

// Keyboard controls
bool eroding = false;

// Programs
std::list<GLuint> programs;

// Models
std::list<Model *> models;

// Camera
glm::vec3 focusPos = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 rCamPos = glm::vec3(0.0f, 1.0f,  1.0f);
float camDistance = 100.0f;


// === CALLBACKS ===

void mouseCallback(GLFWwindow* window, double xpos, double ypos)
{ }

void scrollCallback(GLFWwindow* window, double xOff, double yOff)
{
	camDistance -= yOff*5;
	camDistance = camDistance<50 ? 50 : camDistance;
}

void resizeCallback(GLFWwindow *window, int width, int height)
{
	viewWidth=width, viewHeight=height;
    glViewport(0, 0, width, height);
}

// === FUNCTIONS ===

void processInput(GLFWwindow *window, float deltaTime, glm::vec3 cameraDirection)
{
	// Close window
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
	
	// Generate movement vectors
	float speed = 100.0f * deltaTime;
	glm::vec3 right = speed * glm::normalize( glm::cross(cameraDirection, WORLD_UP) );
	glm::vec3 forward = speed * glm::normalize( glm::cross(WORLD_UP, right) );

	// Erode
    eroding = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;

	// Move
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        focusPos += forward;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        focusPos -= forward;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        focusPos -= right;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        focusPos += right;

	// Turn
	const float SENSITIVITY = 100.0f;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        yaw += SENSITIVITY * deltaTime;
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        yaw -= SENSITIVITY * deltaTime;

	// Constrain pitch
	pitch = pitch >  89.0f ?  89.0f : pitch;
	pitch = pitch < -89.0f ? -89.0f : pitch;
}

Display::Display()
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

	// Setup input
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetCursorPosCallback(window, mouseCallback);
	glfwSetScrollCallback(window, scrollCallback);

	// Load OpenGL
	if ( gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) == 0 )
		throw "GLAD loading error";

	// Set viewport to full window and configure resize callback
	glViewport(0, 0, viewWidth, viewHeight);
	glfwSetFramebufferSizeCallback(window, resizeCallback);

	// Enable Z-Buffering
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Set background colour
	glClearColor(0.3f, 0.7f, 0.9f, 1.0f);
}

void Display::refresh( float currentTime, float deltaTime )
{
	// Calculate camera position
	rCamPos.x = sin( glm::radians(yaw) );
	rCamPos.y = camDistance / 100;
	rCamPos.z = cos( glm::radians(yaw) );
	rCamPos = camDistance * glm::normalize(rCamPos);

	// Move
	processInput( window, deltaTime, glm::normalize(-rCamPos) );

	// Clear screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Common matrices
	glm::mat4 view = glm::lookAt( focusPos+rCamPos, focusPos, WORLD_UP );
	glm::mat4 projection = glm::perspective( glm::radians(60.0f), (float)viewWidth/(float)viewHeight, 0.1f, 1000.0f ); // Clip 10cm - 1km

	// Update shader uniforms
	for( GLuint program : programs )
	{
		glUseProgram(program);
		glUniform1f(		glGetUniformLocation(program, "time"),       currentTime);
		glUniformMatrix4fv(	glGetUniformLocation(program, "view"),       1, GL_FALSE, glm::value_ptr(view) );
		glUniformMatrix4fv(	glGetUniformLocation(program, "projection"), 1, GL_FALSE, glm::value_ptr(projection) );
		glUniform3fv(		glGetUniformLocation(program, "focus"),      1, glm::value_ptr(focusPos) );
		glUseProgram(0);
	}

	// Render models
	for( Model *model : models )
		model->render( glm::vec3(0,0,0) );

	// Check inputs and display
	glfwSwapBuffers(window);
	glfwPollEvents();
}

void Display::addModel(Model &model)
{
	models.push_back(&model);
}

void loadShaders(const char *vertFilePath, const char *fragFilePath, GLuint &programID )
{
	// Create shaders
	GLuint vertShaderID = glCreateShader(GL_VERTEX_SHADER), fragShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Buffer data
	std::ifstream vertShaderStream(vertFilePath, std::ios::in), fragShaderStream(fragFilePath, std::ios::in);
	std::stringstream vsstr, fsstr;
	vsstr << vertShaderStream.rdbuf();
	fsstr << fragShaderStream.rdbuf();
	vertShaderStream.close();
	fragShaderStream.close();

	// Parse data
	std::string vertShaderCode = vsstr.str(), fragShaderCode = fsstr.str();
	const char *vertSourcePointer = vertShaderCode.c_str();
	const char *fragSourcePointer = fragShaderCode.c_str();

	// Compile Shaders
	glShaderSource(vertShaderID, 1, &vertSourcePointer , NULL);
	glShaderSource(fragShaderID, 1, &fragSourcePointer , NULL);
	glCompileShader(vertShaderID);
	glCompileShader(fragShaderID);

	// Check compilation
	GLint vertSucc=GL_FALSE, fragSucc=GL_FALSE;
	glGetShaderiv(vertShaderID, GL_COMPILE_STATUS, &vertSucc);
	glGetShaderiv(fragShaderID, GL_COMPILE_STATUS, &fragSucc);
	if (!vertSucc)
	{
        char infoLog[512];
        glGetShaderInfoLog(vertShaderID, 512, NULL, infoLog);
		Log::print(Log::error, infoLog);
		throw "Vert compilation error";
	}
    if (!fragSucc)
	{
        char infoLog[512];
        glGetShaderInfoLog(fragShaderID, 512, NULL, infoLog);
		Log::print(Log::error, infoLog);
		throw "Frag compilation error";
	}

	// Link program
	programID = glCreateProgram();
	glAttachShader(programID, vertShaderID);
	glAttachShader(programID, fragShaderID);
	glLinkProgram(programID);

	// Check linking
	GLint progSucc=GL_FALSE;
	glGetProgramiv(programID, GL_LINK_STATUS, &progSucc);
	if (!progSucc)
	{
        char infoLog[512];
        glGetProgramInfoLog(programID, 512, NULL, infoLog);
		Log::print(Log::error, infoLog);
		throw "Program linking error";
	}
	
	// Delete shader objects
	glDetachShader(programID, vertShaderID);
	glDetachShader(programID, fragShaderID);
	glDeleteShader(vertShaderID);
	glDeleteShader(fragShaderID);
}

GLuint Display::addShaderProg(const char *vertFilePath, const char *fragFilePath)
{
	GLuint prog;
	loadShaders(vertFilePath, fragFilePath, prog);
	programs.push_back(prog);
	return prog;
}

bool Display::shouldClose()
{
	return glfwWindowShouldClose(window);
}
