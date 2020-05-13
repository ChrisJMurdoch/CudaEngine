
#pragma once

#include <glad/glad.h>

/** Abstract mesh model */
class Model
{
protected:
    // Attribute data
    static const int STRIDE = 6*sizeof(float);
    static const long long ATTR_COORDS = 0*sizeof(float);
    static const long long ATTR_COLOUR = 3*sizeof(float);

    // Mesh data
    GLenum usage;
    int nVertices;
    GLuint VAO, VBO;

public:
    Model(int nVertices, GLenum usage);
    virtual void render() = 0;
};
