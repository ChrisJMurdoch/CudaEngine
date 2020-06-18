
#pragma once

#include "..\..\include\models\model.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class Display
{
public:

    Display();
    void start();
    GLuint addShader(const char *vertFilePath, const char *fragFilePath);
    void addModel(Model &model);
};