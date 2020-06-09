
#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

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

    // Model data
    glm::mat4 position;

public:
    // Shader data
    GLuint program;
    
    Model(int nVertices, GLenum usage, GLuint program);
    virtual void render(float time, glm::mat4 view, glm::mat4 projection) = 0;
};
