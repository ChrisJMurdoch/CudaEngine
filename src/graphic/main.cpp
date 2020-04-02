
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

#include "..\..\include\graphic\main.hpp"

// Mostly triangle tutorial code for now, from http://www.opengl-tutorial.org

int main( void )
{
	// Initialise GLFW
	if( !glfwInit() )
	{
		fprintf( stderr, "Failed to initialize GLFW\n" );
		getchar();
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow( 1024, 768, "CudaEngine", NULL, NULL);
	if( window == NULL ){
		fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible.\n" );
		getchar();
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return -1;
	}

	// Capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// Create and compile GLSL program from the shaders
	GLuint programID = LoadShaders( "shaders\\SimpleVertexShader.vert", "shaders\\SimpleFragmentShader.frag" );
	if ( programID == 0 ) return 1;


	static const GLfloat g_vertex_buffer_data[] = { 
		-1.0f, -1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		 0.0f,  1.0f, 0.0f,
	};

	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	do{

		// Clear the screen
		glClear( GL_COLOR_BUFFER_BIT );

		// Use our shader
		glUseProgram(programID);

		// 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(
			0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		// Draw the triangle
		glDrawArrays(GL_TRIANGLES, 0, 3); // 3 indices starting at 0 -> 1 triangle

		glDisableVertexAttribArray(0);

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
		   glfwWindowShouldClose(window) == 0 );

	// Cleanup VBO
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteVertexArrays(1, &VertexArrayID);
	glDeleteProgram(programID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}

GLuint LoadShaders(const char *vertFilePath,const char *fragFilePath)
{
	// Create shaders
	GLuint vertShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Shaders
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

	// Compile Shaders
	char const *vertSourcePointer = vertShaderCode.c_str();
	char const *fragSourcePointer = fragShaderCode.c_str();
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
		std::cout << "Shader compilation error." << std::endl;
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
		std::cout << "Shader linking error." << std::endl;
		return 0;
	}
	
	// Delete shader objects
	glDetachShader(programID, vertShaderID);
	glDetachShader(programID, fragShaderID);
	glDeleteShader(vertShaderID);
	glDeleteShader(fragShaderID);

	return programID;
}
