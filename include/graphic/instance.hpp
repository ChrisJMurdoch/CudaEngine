
#pragma once

#include <graphic/model.hpp>

class Instance
{
protected:
    Model *model;
    glm::vec3 pos;

public:
    Instance( Model *model, glm::vec3 pos );
    void place( glm::vec3 pos );
    void render();
};
