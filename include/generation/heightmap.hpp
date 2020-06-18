
#pragma once

#include "..\..\include\math\mathEngine.hpp"

#include <map>
#include <string>

class Heightmap
{
public:
    float *nodes;
    int nNodes;

public:
    Heightmap( std::map<std::string, std::string> &settings, int width, MathEngine *math );
    void erode(MathEngine *math, int droplets, int radius);
    ~Heightmap();
};
