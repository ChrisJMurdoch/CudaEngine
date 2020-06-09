
#include <glad/glad.h>

#include "..\..\include\models\eModel.hpp"

EModel::EModel(int nVertices, float *vertexData, int nIndices, unsigned int *indexData, GLuint program, GLenum usage) : Model(nVertices, usage, program)
{
	// Initialise member variables
	this->nIndices = nIndices;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	// Bind buffers
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

	// Copy over data
	glBufferData(GL_ARRAY_BUFFER, nVertices*STRIDE, vertexData, usage);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, nIndices*sizeof(unsigned int), indexData, usage);

	// Position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, STRIDE, (void*)ATTR_COORDS);
	glEnableVertexAttribArray(0);

	// Colour attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, STRIDE, (void*)ATTR_COLOUR);
	glEnableVertexAttribArray(1);

	// Unbind buffers
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void EModel::bufferData(float *vertexData, unsigned int *indexData)
{
	if ( usage != GL_DYNAMIC_DRAW && usage != GL_STREAM_DRAW )
		throw "Buffer not rewritable.";
	
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

    glBufferSubData(GL_ARRAY_BUFFER, 0, nVertices*STRIDE, vertexData);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, nIndices*sizeof(unsigned int), indexData);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void EModel::render(float time, glm::mat4 view, glm::mat4 projection, glm::vec3 focus)
{
	// Bind shaders
	glUseProgram(program);

	// Bind uniforms
	glUniform1f(		glGetUniformLocation(program, "time"), time);
	glUniformMatrix4fv(	glGetUniformLocation(program, "view"),       1, GL_FALSE, glm::value_ptr(view) );
	glUniformMatrix4fv(	glGetUniformLocation(program, "projection"), 1, GL_FALSE, glm::value_ptr(projection) );
	glUniformMatrix4fv(	glGetUniformLocation(program, "model"),      1, GL_FALSE, glm::value_ptr(position) );
	glUniform3fv(		glGetUniformLocation(program, "focus"),      1, glm::value_ptr(focus) );

	// Render
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, nIndices, GL_UNSIGNED_INT, 0);

	// Cleanup
    glBindVertexArray(0);
	glUseProgram(0);
}

EModel::~EModel()
{
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);
}
