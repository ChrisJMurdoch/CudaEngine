
#include "..\..\include\math\cpuMathEngine.hpp"

// GLM
#include <glm/glm.hpp>

// Host code
namespace cpucommon
{
    #include "..\..\src\math\common.cu"
}

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
