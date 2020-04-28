
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

const int VIEW_WIDTH = 800;
const int VIEW_HEIGHT = 600;

int main()
{
	GLFWwindow *window = NULL;
	GLuint programID;

	// Initialise graphics
	if ( initialise(window, programID) != 0 )
	{
		Log::print(Log::error, "Initialisation error.");
		glfwTerminate();
		return -1;
	}

	// Set background
	glClearColor(0.5f, 0.5f, 0.5f, 0.0f);

	// Create triangles
	float vertices[] = {
		// Position - Colour
		0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,
		0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,
		-0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,
		-0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,
	};

	unsigned int indices[] = {
		0, 1, 2,
		2, 3, 0
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

	while( !glfwWindowShouldClose(window) )
	{
		// Get key presses
		processInput(window);

		// Clear screen
		glClear(GL_COLOR_BUFFER_BIT);

		// Use program shaders
		glUseProgram(programID);

		// Bind VAO
		glBindVertexArray(VAO);

		// TRANSFORMATIONS
		// Model
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));

		// View
		glm::mat4 view = glm::mat4(1.0f);
		view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));

		// Projection
		glm::mat4 projection;
		projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

		// UNIFORMS
		// Model
		int modelLoc = glGetUniformLocation(programID, "model");
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

		// View
		modelLoc = glGetUniformLocation(programID, "view");
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(view));

		// Projection
		modelLoc = glGetUniformLocation(programID, "projection");
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(projection));

		// DRAW
		// Draw the triangle
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

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

void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}
