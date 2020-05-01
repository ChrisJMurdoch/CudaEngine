
#define _USE_MATH_DEFINES
#include <cmath>

#include "..\..\include\models\terrain.hpp"

namespace terrain
{
    float modSin(float x, float period)
    {
        return ( sin( x * (2*M_PI) / period ) + 1 ) / 2;
    }
    
    float *generateHeightMap(int width, float min, float max, float period)
    {
        srand(1);
        int n = pow(width, 2);
        float *terrain = new float[n];

        for (int y=0; y<width; y++)
        {
            for (int x=0; x<width; x++)
            {
                int i = y*width + x;

                if ( y==0 || y==width-1 || x==0 || x==width-1 )
                {
                    terrain[i] = 0;
                    continue;
                }

                float x31   = modSin(x, 31*period);
                float y31   = modSin(y, 31*period);
                float x73   = modSin(x, 73*period);
                float y73   = modSin(y, 73*period);

                float a43 = modSin(x+y, 43*period);
                float s43 = modSin(x-y, 43*period);
                float a113 = modSin(x+y, 113*period);
                float s113 = modSin(x-y, 113*period);

                float spike = (float)rand() / (float)RAND_MAX;

                float blend = (
                    10 * x31*y31 +
                    40 * x73*y73 +
                    10 * a43*s43 +
                    39.5 * a113*s113 +
                    .5 * spike
                ) / 100;

                terrain[i] = min + ( (max-min) * blend );
            }
        }
        return terrain;
    }

    float *generateWaterMap(int width, float min, float max, unsigned int seed)
    {
        srand(seed);
        int n = pow(width, 2);
        float *terrain = new float[n];

        for (int y=0; y<width; y++)
        {
            for (int x=0; x<width; x++)
            {
                int i = y*width + x;
                if ( y==0 || y==width-1 || x==0 || x==width-1 )
                {
                    terrain[i] = 0;
                    continue;
                }
                float spike = (float)rand() / (float)RAND_MAX;
                terrain[i] = min + ( (max-min) * spike );
            }
        }
        return terrain;
    }
}
