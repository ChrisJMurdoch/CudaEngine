
#pragma once

namespace terrain
{
    float sinXY(int x, int y, float period);
    float hashXY(int x, int y, float period);
    float *generateHeightMap(int width, float min, float max, float *out, float (*func)(int, int, float), float period = 10);
}
