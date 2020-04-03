
#include <string>
#include <fstream>
#include <sstream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "..\..\include\graphic\main.hpp"
#include "..\..\include\logger\log.hpp"

#define VIEW_WIDTH 800
#define VIEW_HEIGHT 600

int main()
{
	// Initialise GLFW
	glfwInit();

	// Supersample
	glfwWindowHint(GLFW_SAMPLES, 4);
	// OpenGL Version
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	// Force modern GL
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	// Mac compatibility
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	// Create window
	GLFWwindow *window = glfwCreateWindow( VIEW_WIDTH, VIEW_HEIGHT, "CudaEngine", NULL, NULL);
	if (window == NULL)
	{
		Log::print(Log::error, "GLFW window creation error.");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLAD
	if ( !gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) )
	{
		Log::print(Log::error, "GLAD loading error.");
		glfwTerminate();
		return -1;
	}

	// Set viewport and configure resize callback
	glViewport(0, 0, VIEW_WIDTH, VIEW_HEIGHT);
	glfwSetFramebufferSizeCallback(window, resizeCallback);

	// Setup key press mode
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	// Set background
	glClearColor(0.5f, 0.5f, 0.5f, 0.0f);

	// Create and compile GLSL program from the shaders
	GLuint programID = LoadShaders( "shaders\\SimpleVertexShader.vert", "shaders\\SimpleFragmentShader.frag" );
	if (programID == 0)
	{
		Log::print(Log::error, "Shader loading error.");
		glfwTerminate();
		return -1;
	}

	// Create triangle coords
	float vertices[] = {
		-0.5f, -0.5f, 0.0f,
		0.5f, -0.5f, 0.0f,
		0.0f,  0.5f, 0.0f
	};  

	// Create buffer and array objects
	GLuint VAO, VBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	// Bind objects
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	// Copy over buffer data
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// Give OpenGL vertex buffer format
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	// Unbind buffer and array
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	
	while( !glfwWindowShouldClose(window) )
	{
		// Get key presses
		processInput(window);

		// Clear screen
		glClear(GL_COLOR_BUFFER_BIT);

		// Use program shaders
		glUseProgram(programID);

		// Use VAO
		glBindVertexArray(VAO);

		// Draw the triangle
		glDrawArrays(GL_TRIANGLES, 0, 3);

		// Check inputs and display
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Cleanup
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteProgram(programID);

	glfwTerminate();

	return 0;
}

void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void resizeCallback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
}

GLuint LoadShaders(const char *vertFilePath, const char *fragFilePath)
{
	// Create shaders
	GLuint vertShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Shader files
	std::string vertShaderCode, fragShaderCode;
	std::ifstream vertShaderStream(vertFilePath, std::ios::in);
	std::ifstream fragShaderStream(fragFilePath, std::ios::in);
	std::stringstream vsstr, fsstr;
	vsstr << vertShaderStream.rdbuf();
	fsstr << fragShaderStream.rdbuf();
	vertShaderCode = vsstr.str();
	fragShaderCode = fsstr.str();
	vertShaderStream.close();
	fragShaderStream.close();
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
	if (!vertSucc || !fragSucc)
	{
		Log::print(Log::error, "Shader compilation error.");
		return 0;
	}

	// Link the program
	GLuint programID = glCreateProgram();
	glAttachShader(programID, vertShaderID);
	glAttachShader(programID, fragShaderID);
	glLinkProgram(programID);

	// Check linking
	GLint progSucc=GL_FALSE;
	glGetProgramiv(programID, GL_LINK_STATUS, &progSucc);
	if (!progSucc)
	{
		Log::print(Log::error, "Shader linking error.");
		return 0;
	}
	
	// Delete shader objects
	glDetachShader(programID, vertShaderID);
	glDetachShader(programID, fragShaderID);
	glDeleteShader(vertShaderID);
	glDeleteShader(fragShaderID);

	return programID;
}
