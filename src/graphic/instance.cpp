
#include <graphic/instance.hpp>

Instance::Instance( Model *model, glm::vec3 pos )
{
    this->model = model;
    this->pos = pos;
}

void Instance::place( glm::vec3 pos )
{
    this->pos = pos;
}

void Instance::render()
{
    model->render( pos );
}
