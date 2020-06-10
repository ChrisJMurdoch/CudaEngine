
#include "..\..\include\math\cpuMathEngine.hpp"

#include "..\..\include\logger\log.hpp"

// GLM
#include <glm/glm.hpp>

// Host code
namespace cpucommon
{
    #include "..\..\src\math\common.cu"
}

#include <cmath>

// FUNCTIONS

void CPUMathEngine::generateHeightMap(float *out, int dimension, float min, float max, Sample sample, float period, int octaves)
{
    for (int y=0; y<dimension; y++) for (int x=0; x<dimension; x++)
    {
        float value;
        // Custom sampling
        switch ( sample )
        {
        case mountain:
            value = cpucommon::mountain(x, y, period);
            break;
        default:
            value = cpucommon::fractal(x, y, period, sample, octaves);
            break;
        }
        out[y*dimension + x] = min + ( value * (max-min) );
    }
}

float getCellHeight(float *map, int width, int x, int y)
{
    return ( x<0 || x>=width || y<0 || y>=width ) ? 99999999 : map[ x + (y*width) ];
}

void erodeCell(float *map, int width, int x, int y, float load)
{
    // Deposit load
    map[ x + (y*width) ] += 0;

    // Pick next cell
    int lx=0, ly=0;
    float lh = getCellHeight(map, width, x, y);
    for (int xo=-1; xo<2; xo++) for (int yo=-1; yo<2; yo++)
    {
        // Self
        if ( xo==0 && yo==0 ) continue;

        float h = getCellHeight(map, width, x+xo, y+yo);
        if ( h<lh )
        {
            lx = xo;
            ly = ly;
            lh = h;
        }
    }

    // Self is lowest
    if ( lx==0 && ly==0 )
        return;
    
    // Take dirt
    load =  ( getCellHeight(map, width, x, y) - lh ) / 10;
    map[ x + (y*width) ] -= load;

    erodeCell(map, width, x+lx, y+ly, load);
}

void CPUMathEngine::erode(float *map, int width, int droplets)
{
    srand(0);
    for (int i=0; i<droplets; i++)
    {
        int index = rand() % (width*width);
        erodeCell(map, width, index%width, index/width, 0);
    }
}
