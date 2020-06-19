
#pragma once

#include <glad/glad.h>

#include "..\..\include\graphic\model.hpp"
#include "..\..\include\generation\mesh.hpp"

/** VBO mesh model */
class VModel : public Model
{
public:
    VModel(Mesh &mesh, GLuint program, GLenum usage = GL_STATIC_DRAW);
    void bufferData(float *vertexData);
    void render( glm::vec3 position ) override;
    ~VModel();
};
