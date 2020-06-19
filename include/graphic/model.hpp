
#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

/** Purely visual OpenGL mesh model */
class Model
{
protected:
    // Vertex attribute data
    static const int STRIDE = 6*sizeof(float);
    static const long long ATTR_COORDS = 0*sizeof(float);
    static const long long ATTR_COLOUR = 3*sizeof(float);

    // Mesh data
    GLenum usage;
    int nVertices;
    GLuint VAO, VBO;

public:
    // Shader data
    GLuint program;
    
    Model(int nVertices, GLuint program, GLenum usage);

    virtual void render( glm::vec3 position ) = 0;
};
