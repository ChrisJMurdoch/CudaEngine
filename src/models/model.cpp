
#include <glad/glad.h>

#include "..\..\include\models\model.hpp"

Model::Model(float *vertexData, int nVertices)
{
    this->nVertices = nVertices;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
    setVertexData(vertexData);
}

void Model::setVertexData(float *vertexData)
{
	// Bind buffers
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	// Copy over data
	glBufferData(GL_ARRAY_BUFFER, nVertices*STRIDE, vertexData, GL_STATIC_DRAW);
    

	// Position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, STRIDE, (void*)ATTR_COORDS);
	glEnableVertexAttribArray(0);

	// Colour attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, STRIDE, (void*)ATTR_COLOUR);
	glEnableVertexAttribArray(1);

	// Unbind buffers
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Model::render()
{
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, nVertices);
    glBindVertexArray(0);
}

Model::~Model()
{
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
}
