
#pragma once

/** Maths engine interface to abstract cpu and gpu implementations */
class MathEngine
{
public:

    /** Type of point sampling for heightmap generation */
    enum Sample { hash, sin, perlin };

    /** Create heightmap on gpu */
    virtual void generateHeightMap(int dimension, float min, float max, float *out, Sample sample = sin, float period = 10, int octaves = 1) = 0;
};
