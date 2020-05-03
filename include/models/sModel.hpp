
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
public:
    SModel(float *vertexData, int nVertices, bool dynamic=false);
    void render();
    ~SModel();
};
