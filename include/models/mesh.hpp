
#pragma once

namespace mesh
{
    float *generateVertices(float *heightmap, int width);
    void setVertex(float *vertex, float x, float y, float z, float flat);
}
