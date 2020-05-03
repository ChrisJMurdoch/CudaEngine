
#include <glad/glad.h>

#include "..\..\include\models\dModel.hpp"
#include "..\..\include\logger\log.hpp"

DModel::DModel(float *vertexData, int nVertices) : SModel(vertexData, nVertices, true) {}

void DModel::setVertexData(float *vertexData, int nVertices)
{
	// Bind buffer
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	// Copy over data
    glBufferSubData(GL_ARRAY_BUFFER, 0, nVertices*STRIDE, vertexData);

	// Unbind buffers
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}
