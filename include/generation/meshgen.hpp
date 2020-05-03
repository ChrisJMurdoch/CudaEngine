
#pragma once

namespace meshgen
{
    enum ColourScheme { landscape, water };
    float *generateVertices(float *heightmap, int width, float *out, ColourScheme cs);
    void setVertex(float *vertex, float x, float y, float z, float flat, float triY, ColourScheme cs);
    void setColour(float *vertex, float flat, glm::vec3 steepCol, glm::vec3 flatCol, float min=0, float max=1);
}
