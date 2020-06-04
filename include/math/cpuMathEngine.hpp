
#pragma once

#include "..\..\include\math\mathEngine.hpp"

class CPUMathEngine : public MathEngine
{
public:

    /** Create heightmap on cpu */
    void generateHeightMap(float *out, int dimension, float min, float max, Sample sample, float period, int octaves=1) override;
};
