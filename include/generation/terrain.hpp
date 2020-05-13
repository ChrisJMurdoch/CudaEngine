
#pragma once

namespace terrain
{
    float *generateHeightMap(int width, float min, float max, float *out, float period=1);
    float *generateWaterMap(int width, float min, float max, float *out);
    float *generateMovingWaterMap(int width, float min, float max, float *out, float waveheight, float time);
}
