
#include <glad/glad.h>

#include "..\..\include\graphic\model.hpp"

Model::Model(int nVertices, GLuint program, GLenum usage)
{
    this->nVertices = nVertices;
    this->program = program;
	this->usage = usage;
}
