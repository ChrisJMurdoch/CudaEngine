
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

void brush(float *map, int width, int x, int y, float amount, int radius)
{
    int dim = (radius*2)+1;
    amount /= (float)dim*dim;

    // Calc distribution
    float total = 0;
    for (int xo=-radius; xo<radius+1; xo++) for (int yo=-radius; yo<radius+1; yo++)
    {
        // Out of bounds
        if  ( x+xo<0 || x+xo>=width || y+yo<0 || y+yo>=width )
            continue;
        
        // Fade
        float dist = pow( pow( (xo), 2) + pow( (yo), 2), 0.5);
        float close = (radius - dist) / radius * 2;
        close = close<0 ? 0 : close;

        total += close;
    }
    float mult = dim*dim / total;

    // Add values
    for (int xo=-radius; xo<radius+1; xo++) for (int yo=-radius; yo<radius+1; yo++)
    {
        // Out of bounds
        if  ( x+xo<0 || x+xo>=width || y+yo<0 || y+yo>=width )
            continue;

        // Fade
        float dist = pow( pow( (xo), 2) + pow( (yo), 2), 0.5);
        float close = (radius - dist) / radius * 2;
        close = close<0 ? 0 : close;

        // Paint
        map[ x+xo + ((y+yo)*width) ] += mult*amount*close;
        total++;
    }
}

float getCellHeight(float *map, int width, int x, int y)
{
    return ( x<0 || x>=width || y<0 || y>=width ) ? 9999 : map[ x + (y*width) ]+1000;
}

void erodeCell(float *map, int width, int x, int y, float speed, float sediment, int radius)
{
    // Calc height
    float height = getCellHeight(map, width, x, y);

    // Find lowest cell in 3x3
    int lx=x, ly=y;
    float lh = height;
    for (int xo=-1; xo<2; xo++) for (int yo=-1; yo<2; yo++)
    {
        float h = getCellHeight(map, width, x+xo, y+yo);
        if ( h<lh )
        {
            lx = x+xo;
            ly = y+yo;
            lh = h;
        }
    }

    // Calculate difference
    float delta = height - lh;

    // Stuck
    if ( -delta >= speed )
    {
        brush(map, width, x, y, sediment, radius);
        return;
    }

    // CUSTOM
    speed += delta-0.1f;
    float capacity = speed;
    
    // Alter sediment
    float deposit = sediment - capacity;
    brush(map, width, x, y, deposit, radius);

    erodeCell(map, width, lx, ly, speed, capacity, radius);
}

void CPUMathEngine::erode(float *map, int width, int droplets, int radius)
{
    static float s_i = 0;
    srand( s_i++ );
    for (int i=0; i<droplets; i++)
    {
        int x = rand() % width, y = rand() % width;
        erodeCell(map, width, x, y, 0, 0, radius);
    }
}
