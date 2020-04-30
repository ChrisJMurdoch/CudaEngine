
#include <cmath>

#include "..\..\include\models\terrain.hpp"

namespace terrain
{
    float *generateHeightMap(int width, float min, float max, unsigned int seed)
    {
        srand(seed);
        int n = pow(width, 2);
        float *terrain = new float[n];
        for (int i=0; i<n; i++)
        {
            float r = (float)rand() / (float)RAND_MAX;
            terrain[i] = ( r * (max-min) ) + min;
        }
        return terrain;
    }
}
