
#pragma once

#include "..\..\include\graphic\model.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class Display
{
public:

    Display();
    void refresh( float currentTime, float deltaTime );
    GLuint addShaderProg(const char *vertFilePath, const char *fragFilePath);
    void addModel(Model &model);
    bool shouldClose();
};