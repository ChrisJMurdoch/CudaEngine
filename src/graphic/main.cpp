
// Image loading
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// OpenGL loading
#include <glad/glad.h>

// Window creation
#include <GLFW/glfw3.h>

#include "..\..\include\graphic\main.hpp"
#include "..\..\include\graphic\util.hpp"
#include "..\..\include\logger\log.hpp"

#define VIEW_WIDTH 800
#define VIEW_HEIGHT 600

int main()
{
	// Initialise graphics
	GLFWwindow *window = NULL;
	GLuint programID;
	if ( initialise(&window, &programID) != 0 )
		return -1;

	// Set background
	glClearColor(0.5f, 0.5f, 0.5f, 0.0f);

	// Create triangles
	float vertices[] = {
		// Position-Colour-TexturePos
		0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f,
		0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f,
		-0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,
		-0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f
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
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// Colour attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
	glEnableVertexAttribArray(1);
	// Texture attribute
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	// Unbind buffers and array
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	
	// Create and load textures
    GLuint texture1, texture2;
	loadTexture("assets/textures/container.jpg", GL_RGB, &texture1);
	loadTexture("assets/textures/awesomeface.png", GL_RGBA, &texture2);

    // Set texture positions
    glUseProgram(programID);
    glUniform1i(glGetUniformLocation(programID, "tex1"), 0);
	glUniform1i(glGetUniformLocation(programID, "tex2"), 1);

	while( !glfwWindowShouldClose(window) )
	{
		// Get key presses
		processInput(window);

		// Clear screen
		glClear(GL_COLOR_BUFFER_BIT);

		// Use textures
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D,  texture1);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);

		// Use program shaders
		glUseProgram(programID);

		// Bind VAO
		glBindVertexArray(VAO);

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

int loadTexture(const char *filepath, GLenum rasterType, GLuint *texPtr)
{
	// Create texture
	glGenTextures(1, texPtr);
    glBindTexture(GL_TEXTURE_2D, *texPtr);

    // Settings
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // load image, create texture and generate mipmaps
    int width, height, nChannels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char *data = stbi_load(filepath, &width, &height, &nChannels, 0);
    if (!data)
    {
        Log::print( Log::error, "Texture loading error.");
    	stbi_image_free(data);
		return -1;
    }

	// Load image into texture
	glTexImage2D(GL_TEXTURE_2D, 0, rasterType, width, height, 0, rasterType, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(data);
	return 0;
}

int initialise(GLFWwindow **window, GLuint *programPtr)
{
	// Create window
	if ( createWindow(window) != 0 )
	{
		Log::print(Log::error, "Window creation error.");
		glfwTerminate();
		return -1;
	}

	// Initialise GLAD
	if ( initGLAD(*window) != 0 )
	{
		Log::print(Log::error, "GLAD initialising error.");
		glfwTerminate();
		return -1;
	}

	// Load shaders
	*programPtr = LoadShaders( "shaders\\SimpleVertexShader.vert", "shaders\\SimpleFragmentShader.frag" );
	if (*programPtr == 0)
	{
		Log::print(Log::error, "Shader loading error.");
		glfwTerminate();
		return -1;
	}

	return 0;
}

int createWindow(GLFWwindow **window)
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
	*window = glfwCreateWindow( VIEW_WIDTH, VIEW_HEIGHT, "CudaEngine", NULL, NULL);
	if (*window == NULL)
		return -1;
	glfwMakeContextCurrent(*window);

	// Setup key press mode
	glfwSetInputMode(*window, GLFW_STICKY_KEYS, GL_TRUE);
	
	return 0;
}

int initGLAD(GLFWwindow *window)
{
	// Load GLAD
	if ( !gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) )
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
