
#pragma once

#include "..\..\include\math\mathEngine.hpp"

class CPUMathEngine : public MathEngine
{
public:

    /** Create heightmap on cpu */
    void generateHeightMap(float *out, int dimension, float min, float max, Sample sample, float period, int octaves=1) override;

    /** Erode terrain heightmap */
    void erode(float *map, int width, int droplets, int radius) override;

private:

    void brush(float *map, int width, int x, int y, float amount, int radius);
    float getCellHeight(float *map, int width, int x, int y);
    void erodeCell(float *map, int width, int x, int y, float speed, float sediment, int radius);

};
