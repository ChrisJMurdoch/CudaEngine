
#include <glad/glad.h>

#include "..\..\include\models\vModel.hpp"

VModel::VModel(int nVertices, float *vertexData, GLuint program, GLenum usage) : Model(nVertices, usage, program)
{
	// Initialise member variables
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	// Bind buffers
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	// Copy over data
	glBufferData(GL_ARRAY_BUFFER, nVertices*STRIDE, vertexData, usage);

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

void VModel::bufferData(float *vertexData)
{
	if ( usage != GL_DYNAMIC_DRAW && usage != GL_STREAM_DRAW )
		throw "Buffer not rewritable.";
	
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, nVertices*STRIDE, vertexData);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void VModel::render()
{
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, nVertices);
    glBindVertexArray(0);
}

VModel::~VModel()
{
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
}
