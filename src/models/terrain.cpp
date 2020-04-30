
#define _USE_MATH_DEFINES
#include <cmath>

#include "..\..\include\models\terrain.hpp"

namespace terrain
{
    float modSin(float x, float period)
    {
        return ( sin( x * (2*M_PI) / period ) + 1 ) / 2;
    }

    float *generateHeightMap(int width, float min, float max, unsigned int seed)
    {
        srand(seed);
        int n = pow(width, 2);
        float *terrain = new float[n];

        for (int y=0; y<width; y++)
        {
            for (int x=0; x<width; x++)
            {
                int i = y*width + x;

                float x31   = modSin(x, 31);
                float y31   = modSin(y, 31);
                float x73   = modSin(x, 73);
                float y73   = modSin(y, 73);

                float a43 = modSin(x+y, 43);
                float s43 = modSin(x-y, 43);
                float a113 = modSin(x+y, 113);
                float s113 = modSin(x-y, 113);

                float spike = (float)rand() / (float)RAND_MAX;

                float blend = ( 20*x31*y31 + 30*x73*y73 + 20*a43*s43 + 29*a113*s113 + 0.5*spike ) / 100;

                terrain[i] = min + ( (max-min) * blend );
            }
        }
        return terrain;
    }
}
