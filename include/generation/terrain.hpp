
#pragma once

namespace terrain
{
    float *generateHeightMap(int width, float min, float max, float *out, float period=1);
    float *generateWaterMap(int width, float min, float max, float *out, unsigned int seed=0);
}
