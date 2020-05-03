
#pragma once

class Model
{
private:
    // Attribute data
    const int STRIDE = 6*sizeof(float);
    const long long ATTR_COORDS = 0*sizeof(float);
    const long long ATTR_COLOUR = 3*sizeof(float);
    // Mesh data
    GLuint VAO, VBO;
    int nVertices;
public:
    Model(float *vertexData, int nVertices);
    void setVertexData(float *vertexData);
    void render();
    ~Model();
};
