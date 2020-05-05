
#include <glad/glad.h>

#include "..\..\include\models\dModel.hpp"

DModel::DModel(float *vertexData, int nVertices) : SModel(vertexData, nVertices, GL_STREAM_DRAW) {}

void DModel::setVertexData(float *vertexData, int nVertices)
{
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, nVertices*STRIDE, vertexData);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
