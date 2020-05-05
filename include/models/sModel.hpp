
#pragma once

/** Non-rewritable mesh model */
class SModel
{
protected:
    // Attribute data
    static const int STRIDE = 6*sizeof(float);
    static const long long ATTR_COORDS = 0*sizeof(float);
    static const long long ATTR_COLOUR = 3*sizeof(float);
    // Mesh data
    GLuint VAO, VBO;
    int nVertices;

protected:
    /** To be internally chained with a GL_*_DRAW enum */
    SModel(float *vertexData, int nVertices, GLenum usage);

public:
    /** Chains to protected ctor with GL_STATIC_DRAW */
    SModel(float *vertexData, int nVertices);
    void render();
    ~SModel();
};
