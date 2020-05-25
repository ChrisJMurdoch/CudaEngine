
#pragma once

namespace terrain
{
    float msin(int x, int y, float period);
    float hash(int x, int y, float period);
    float perlin(int x, int y, float period);
    float *generateHeightMap(int width, float min, float max, float *out, float (*func)(int, int, float), float period = 10, int octaves = 1);
}
