
#pragma once

#include "..\..\include\math\mathEngine.hpp"

class CPUMathEngine : public MathEngine
{
public:

    CPUMathEngine();

    /** Create heightmap on cpu */
    void generateHeightMap(int dimension, float min, float max, float *out, Sample sample = sin, float period = 10, int octaves = 1) override;
};
