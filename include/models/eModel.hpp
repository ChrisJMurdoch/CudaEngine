
#pragma once

#include <glad/glad.h>

#include "..\..\include\models\model.hpp"

/** EBO mesh model */
class EModel : public Model
{
private:
    // Mesh data
    int nIndices;
    GLuint EBO;

public:
    EModel(int nVertices, float *vertexData, int nIndices, unsigned int *indexData, GLenum usage = GL_STATIC_DRAW);
    void bufferData(float *vertexData, unsigned int *indexData);
    void render() override;
    ~EModel();
};
