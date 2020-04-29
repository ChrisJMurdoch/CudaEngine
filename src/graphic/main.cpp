

// === INCLUDES ===

// OpenGL initialisation
#include <glad/glad.h>

// Window creation
#include <GLFW/glfw3.h>

// Image loading
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// Vector math
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "..\..\include\graphic\main.hpp"
#include "..\..\include\graphic\util.hpp"
#include "..\..\include\logger\log.hpp"


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
glm::vec3 cameraPosition = glm::vec3(0.0f, 0.0f,  3.0f);
glm::vec3 cameraDirection = glm::vec3(0.0f, 0.0f, -1.0f);


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

	// Model positions
	glm::vec3 modelPositions[] = {
		glm::vec3( 0.0f,  0.0f,  0.0f),
		glm::vec3( 2.0f,  5.0f, -15.0f),
		glm::vec3(-1.5f, -2.2f, -2.5f),
		glm::vec3(-3.8f, -2.0f, -12.3f),
		glm::vec3( 2.4f, -0.4f, -3.5f),
		glm::vec3(-1.7f,  3.0f, -7.5f),
		glm::vec3( 1.3f, -2.0f, -2.5f),
		glm::vec3( 1.5f,  2.0f, -2.5f),
		glm::vec3( 1.5f,  0.2f, -1.5f),
		glm::vec3(-1.3f,  1.0f, -1.5f),
	};

	// Triangles
	float vertices[] = {
		// Front
		-0.5, -0.5, -0.5,   1, 0, 0,
		-0.5,  0.5, -0.5,   1, 0, 0,
		 0.5,  0.5, -0.5,   1, 0, 0,
		 0.5, -0.5, -0.5,   1, 0, 0,
		// Back
		-0.5, -0.5,  0.5,   0, 1, 0,
		-0.5,  0.5,  0.5,   0, 1, 0,
		 0.5,  0.5,  0.5,   0, 1, 0,
		 0.5, -0.5,  0.5,   0, 1, 0,
		 // Top
		-0.5,  0.5, -0.5,   0, 0, 1,
		-0.5,  0.5,  0.5,   0, 0, 1,
		 0.5,  0.5,  0.5,   0, 0, 1,
		 0.5,  0.5, -0.5,   0, 0, 1,
		 // Bottom
		-0.5, -0.5, -0.5,   1, 1, 0,
		-0.5, -0.5,  0.5,   1, 1, 0,
		 0.5, -0.5,  0.5,   1, 1, 0,
		 0.5, -0.5, -0.5,   1, 1, 0,
		 // Left
		-0.5, -0.5, -0.5,   0, 1, 1,
		-0.5, -0.5,  0.5,   0, 1, 1,
		-0.5,  0.5,  0.5,   0, 1, 1,
		-0.5,  0.5, -0.5,   0, 1, 1,
		//Right
		 0.5, -0.5, -0.5,   1, 0, 1,
		 0.5, -0.5,  0.5,   1, 0, 1,
		 0.5,  0.5,  0.5,   1, 0, 1,
		 0.5,  0.5, -0.5,   1, 0, 1,
	};
	unsigned int indices[] = {
		// Front
		0, 1, 2,
		2, 3, 0,
		// Back
		4, 5, 6,
		6, 7, 4,
		// Top
		8, 9, 10,
		10, 11, 8,
		// Bottom
		12, 13, 14,
		14, 15, 12,
		// Left
		16, 17, 18,
		18, 19, 16,
		// Right
		20, 21, 22,
		22, 23, 20,
	}; 

	// Create buffer and array objects
	GLuint VAO, VBO, EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	// Bind objects
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

	// Copy over buffer data
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// Colour attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)(3*sizeof(float)));
	glEnableVertexAttribArray(1);

	// Unbind buffers and array
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	float lastFrame = glfwGetTime();


	while( !glfwWindowShouldClose(window) )
	{
		// Get time-delta
		float currentFrame = glfwGetTime();
		float deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		
		// Calculate view angle
		glm::vec3 direction;
		direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
		direction.y = sin(glm::radians(pitch));
		direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
		glm::vec3 cameraDirection = glm::normalize(direction);

		// Move
		processInput(window, deltaTime, cameraDirection);

		// Clear screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Use program shaders
		glUseProgram(programID);

		// Bind VAO
		glBindVertexArray(VAO);

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
		for(int i=0; i<10; i++)
		{
			// Model position
			glm::mat4 model = glm::mat4(1.0f);
			model = glm::translate(model, modelPositions[i]);
			model = glm::rotate(model, glm::radians(20.0f*i) + (float)glfwGetTime(), glm::vec3(1.0f, 0.3f, 0.5f));

			// Create uniform
			modelLoc = glGetUniformLocation(programID, "model");
			glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

			// Draw
			glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
		}

		// Unbind VAO
		glBindVertexArray(0);

		// Check inputs and display
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Cleanup
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);
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
	glClearColor(0.5f, 0.5f, 0.5f, 0.0f);

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

void resizeCallback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
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
	const float cameraSpeed = 2.0f * deltaTime;
	const glm::vec3 forward = cameraSpeed * cameraDirection;
	const glm::vec3 right   = cameraSpeed * glm::normalize(glm::cross(cameraDirection, WORLD_UP));

	// Move
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPosition += forward;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPosition -= forward;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPosition -= right;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPosition += right;
}
