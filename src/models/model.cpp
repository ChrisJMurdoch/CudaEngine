
#include <glad/glad.h>

#include "..\..\include\models\model.hpp"

Model::Model(int nVertices, GLenum usage)
{
    this->nVertices = nVertices;
	this->usage = usage;
}
