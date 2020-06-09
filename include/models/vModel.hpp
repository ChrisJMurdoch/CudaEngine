
#pragma once

#include <glad/glad.h>

#include "..\..\include\models\model.hpp"

/** VBO mesh model */
class VModel : public Model
{
public:
    VModel(int nVertices, float *vertexData, GLuint program, GLenum usage = GL_STATIC_DRAW);
    void bufferData(float *vertexData);
    void render(float time, glm::mat4 view, glm::mat4 projection, glm::vec3 focus) override;
    ~VModel();
};
