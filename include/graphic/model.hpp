
#pragma once

#include <generation/mesh.hpp>

#include <glad/glad.h>
#include <glm/glm.hpp>

/** VBO mesh model */
class Model
{
private:
    // Mesh data
    GLenum usage;
    int nVertices;
    GLuint VAO, VBO;

    // Shader data
    GLuint program;

public:
    // Vertex attribute data
    static const int VERTEX_STRIDE = 9;
    static const int ATTR_COORDS = 0;
    static const int ATTR_COLOUR = 3;
    static const int ATTR_NORMAL = 6;

public:
    Model(Mesh &mesh, GLuint program, GLenum usage = GL_STATIC_DRAW);
    void bufferData(float *vertexData);
    void render( glm::vec3 position );
    ~Model();
};
