
#include <glad/glad.h>

#include "..\..\include\models\model.hpp"

Model::Model(int nVertices, GLenum usage, GLuint program, glm::vec3 position)
{
    this->nVertices = nVertices;
	this->usage = usage;
    this->program = program;
    this->position = glm::translate( glm::mat4(1.0f), position );
}

void Model::place( glm::vec3 position )
{
    this->position = glm::translate( glm::mat4(1.0f), position );
}
