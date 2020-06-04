
#pragma once

/** Maths engine interface to abstract cpu and gpu implementations */
class MathEngine
{
public:

    /** Type of point sampling for heightmap generation */
    enum Sample { hash, sin, perlin };

    /** Create heightmap on gpu */
    virtual void generateHeightMap(float *out, int dimension, float min, float max, Sample sample, float period, int octaves=1) = 0;
};
