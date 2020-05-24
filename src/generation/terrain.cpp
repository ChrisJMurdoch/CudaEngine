
#define _USE_MATH_DEFINES
#include <functional>
#include <cmath>

#include "..\..\include\generation\terrain.hpp"
#include "..\..\include\logger\log.hpp"

namespace terrain
{
    // SAMPLING

    float sinXY(int x, int y, float period)
    {
        float xd = ( sin( x * (2*M_PI) / period ) + 1 ) / 2;
        float yd = ( sin( y * (2*M_PI) / period ) + 1 ) / 2;
        return xd * yd;
    }

    float hashXY(int x, int y, float period)
    {
        static std::hash<int> hash;
        return ( hash( x*x*y ) % 1001 ) / 1000.0f;
    }

    // GENERATION
    
    float *generateHeightMap(int width, float min, float max, float *out, float (*sampler)(int, int, float), float period)
    {
        int n = pow(width, 2);

        for (int y=0; y<width; y++) for (int x=0; x<width; x++)
        {
            int i = y*width + x;
            out[i] = min + ( (max-min) * sampler(x, y, period) );
        }
        
        return out;
    }
}
