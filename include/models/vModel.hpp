
#pragma once

#include <glad/glad.h>

#include "..\..\include\models\model.hpp"

/** VBO mesh model */
class VModel : Model
{
public:
    VModel(int nVertices, float *vertexData, GLenum usage = GL_STATIC_DRAW);
    void bufferData(float *vertexData);
    void render() override;
    ~VModel();
};
