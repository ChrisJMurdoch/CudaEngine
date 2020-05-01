
#pragma once

namespace mesh
{
    float *generateVertices(float *heightmap, int width, bool water);
    void setVertex(float *vertex, float x, float y, float z, float flat, float triY, bool water);
    void setColour(float *vertex, float flat, glm::vec3 steepCol, glm::vec3 flatCol, float delay=0);
}
