
#pragma once

#include <graphic/model.hpp>

#include <generation/mesh.hpp>

#include <glad/glad.h>
#include <glm/glm.hpp>

/** VBO mesh model */
class Model
{
private:
    // Vertex attribute data
    static const int STRIDE = 6*sizeof(float);
    static const long long ATTR_COORDS = 0*sizeof(float);
    static const long long ATTR_COLOUR = 3*sizeof(float);

    // Mesh data
    GLenum usage;
    int nVertices;
    GLuint VAO, VBO;

    // Shader data
    GLuint program;

public:
    Model(Mesh &mesh, GLuint program, GLenum usage = GL_STATIC_DRAW);
    void bufferData(float *vertexData);
    void render( glm::vec3 position );
    ~Model();
};
