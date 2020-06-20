
#include <graphic/model.hpp>

#include <glm/gtc/type_ptr.hpp>

Model::Model(Mesh &mesh, GLuint program, GLenum usage)
{
	this->nVertices = mesh.nVertices;
    this->program = program;
	this->usage = usage;

	// Initialise member variables
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	// Bind buffers
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	// Copy over data
	glBufferData(GL_ARRAY_BUFFER, nVertices*VERTEX_STRIDE*sizeof(float), mesh.vertexData, usage);

	// Position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE*sizeof(float), (void*)(ATTR_COORDS*sizeof(float)));
	glEnableVertexAttribArray(0);

	// Colour attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE*sizeof(float), (void*)(ATTR_COLOUR*sizeof(float)));
	glEnableVertexAttribArray(1);

	// Normal attribute
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE*sizeof(float), (void*)(ATTR_NORMAL*sizeof(float)));
	glEnableVertexAttribArray(2);

	// Unbind buffers
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Model::bufferData(float *vertexData)
{
	if ( usage != GL_DYNAMIC_DRAW && usage != GL_STREAM_DRAW )
		throw "Buffer not rewritable.";
	
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, nVertices*VERTEX_STRIDE*sizeof(float), vertexData);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Model::render( glm::vec3 position )
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

Model::~Model()
{
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
}
