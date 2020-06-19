
#pragma once

#include <graphic/model.hpp>

#include <glad/glad.h>

class Display
{
public:
    Display();
    void refresh( float currentTime, float deltaTime );
    GLuint addShaderProg(const char *vertFilePath, const char *fragFilePath);
    void addModel(Model &model);
    bool shouldClose();
};
