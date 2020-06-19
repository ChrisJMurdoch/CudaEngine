
#include <generation/heightmap.hpp>

Heightmap::Heightmap( std::map<std::string, std::string> &settings, int width, MathEngine *math )
{
    nNodes = width*width;
    nodes = new float[nNodes];

    // Parse sampling to enum
    MathEngine::Sample sample;
    const char *s = settings["sampling"].c_str();
    if ( strcmp(s, "hash") == 0 )
        sample = MathEngine::hash;
    else if ( strcmp(s, "perlin") == 0 )
        sample = MathEngine::perlin;
    else if ( strcmp(s, "perlinRidge") == 0 )
        sample = MathEngine::perlinRidge;
    else if ( strcmp(s, "mountain") == 0 )
        sample = MathEngine::mountain;
    else
        sample = MathEngine::hash;
    
    math->generateHeightMap(nodes, width, stof(settings["min"]), stof(settings["max"]), sample, stof(settings["period"]), stoi(settings["octaves"]));
}

void Heightmap::erode(MathEngine *math, int droplets, int radius)
{
    math->erode(nodes, pow( nNodes, 0.5 ), droplets, radius);
}

Heightmap::~Heightmap()
{
    delete[] nodes;
}
