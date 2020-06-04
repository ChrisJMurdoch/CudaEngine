
// GLM vector maths
#include <glm/glm.hpp>

#include "..\..\include\math\cpuMathEngine.hpp"

namespace cpucommon
{
    #include "..\..\src\math\common.cu"
}

#define M_PI 3.14159265358979323846

// DEVICE SETUP

CPUMathEngine::CPUMathEngine() {}

// FUNCTIONS

void CPUMathEngine::generateHeightMap(float *out, int dimension, float min, float max, Sample sample, float period, int octaves)
{

    const float lacunarity = 2;
    const float persistance = 0.4;

    int n = pow(dimension, 2);

    for (int y=0; y<dimension; y++) for (int x=0; x<dimension; x++)
    {
        int i = y*dimension + x;

        float height = 0;
        for (int o=0; o<octaves; o++)
        {
            float ol = pow(lacunarity, o);
            float op = pow(persistance, o);

            float value;
            switch ( sample )
            {
            case hash:
                value = cpucommon::hashSample( x, y, period );
                break;
            case sin:
                value = cpucommon::sinSample( x, y, period );
                break;
            case perlin:
                value = cpucommon::perlinSample( x, y, period );
                break;
            default:
                value = cpucommon::hashSample( x, y, period );
                break;
            }
            height += ( min + ( (max-min) * value ) ) * op;

            if ( ol != 1 )
                height -= ( min + ( (max-min) * 0.5 ) ) * op;

            // Dropoff
            float unitX = (float)x / dimension, unitY = (float)y / dimension;
            height *= cpucommon::falloff(unitX) * cpucommon::falloff(unitY);
        }

        out[i] = height;
    }
}
