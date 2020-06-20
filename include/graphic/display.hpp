
#pragma once

#include <graphic/instance.hpp>

#include <glad/glad.h>

class Display
{
public:
    Display();
    void refresh( float currentTime, float deltaTime );
    GLuint addShaderProg(const char *vertFilePath, const char *fragFilePath);
    void addInstance(Instance *instance);
    bool shouldClose();
};
