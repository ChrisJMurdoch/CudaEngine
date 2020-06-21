
#pragma once

#include <generation/heightmap.hpp>

class Mesh
{
public:
    enum ColourScheme { landscape, water };

    float *vertexData;
    int nVertices;
    
public:
    Mesh( float *vertexData, int nVertices );
    Mesh( Heightmap &primary, ColourScheme cs );
    Mesh( Heightmap &primary, Heightmap &compare, ColourScheme cs );
    ~Mesh();
};
