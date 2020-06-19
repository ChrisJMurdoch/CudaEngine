
#include <graphic/vModel.hpp>

#include <glm/gtc/type_ptr.hpp>

VModel::VModel(Mesh &mesh, GLuint program, GLenum usage) : Model(mesh.nVertices, program, usage)
{
	// Initialise member variables
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	// Bind buffers
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	// Copy over data
	glBufferData(GL_ARRAY_BUFFER, nVertices*STRIDE, mesh.vertexData, usage);

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

void VModel::render( glm::vec3 position )
{
	// Bind shaders
	glUseProgram(program);                   

	// Bind uniform
	glUniformMatrix4fv(	glGetUniformLocation(program, "model"), 1, GL_FALSE, glm::value_ptr( glm::translate( glm::mat4(1.0f), position ) ) );

	// Render
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, nVertices);

	// Cleanup
    glBindVertexArray(0);
	glUseProgram(0);
}

VModel::~VModel()
{
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
}
