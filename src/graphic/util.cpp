
#include "..\..\include\graphic\util.hpp"

#include "..\..\include\logger\log.hpp"

#include <string>
#include <fstream>
#include <sstream>

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
