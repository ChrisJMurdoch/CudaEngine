
#include <glad/glad.h>

#include "..\..\include\models\model.hpp"

Model::Model(int nVertices, GLenum usage, GLuint program)
{
    this->nVertices = nVertices;
	this->usage = usage;
    this->program = program;
    position = glm::mat4(1.0f);
}
