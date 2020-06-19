
#pragma once

#include <generation\heightmap.hpp>

class Mesh
{
public:
    static const int VERTEX_SIZE = 6;
    static const int COORD_INDEX = 0;
    static const int COLOUR_INDEX = 3;

    enum ColourScheme { landscape, water };

    float *vertexData;
    int nVertices;
    
public:
    Mesh( Heightmap &primary, ColourScheme cs );
    Mesh( Heightmap &primary, Heightmap &compare, ColourScheme cs );
    ~Mesh();
};
