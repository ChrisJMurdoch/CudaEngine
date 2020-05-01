
#pragma once

namespace terrain
{
    float *generateHeightMap(int width, float min, float max, float period=1);
    float *generateWaterMap(int width, float min, float max, unsigned int seed=0);
}
