
#include <glad/glad.h>

#include "..\..\include\models\sModel.hpp"
#include "..\..\include\logger\log.hpp"

SModel::SModel(float *vertexData, int nVertices, bool dynamic)
{
    this->nVertices = nVertices;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	// Bind buffers
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	// Copy over data
	if (dynamic)
		glBufferData(GL_ARRAY_BUFFER, nVertices*STRIDE, vertexData, GL_DYNAMIC_DRAW);
	else
		glBufferData(GL_ARRAY_BUFFER, nVertices*STRIDE, vertexData, GL_STATIC_DRAW);

	// Position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, STRIDE, (void*)ATTR_COORDS);
	glEnableVertexAttribArray(0);

	// Colour attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, STRIDE, (void*)ATTR_COLOUR);
	glEnableVertexAttribArray(1);

	// Unbind buffer
	glBindVertexArray(0);
}

void SModel::render()
{
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, nVertices);
    glBindVertexArray(0);
}

SModel::~SModel()
{
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
}
