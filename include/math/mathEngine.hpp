
#pragma once

/** Maths engine interface to abstract cpu and gpu implementations */
class MathEngine
{
public:

    /** Type of point sampling for heightmap generation */
    enum Sample { hash, sin, perlin, perlinRidge, mountain };

    /** Create heightmap */
    virtual void generateHeightMap(float *out, int dimension, float min, float max, Sample sample, float period, int octaves=1) = 0;

    /** Erode terrain heightmap */
    virtual void erode(float *map, int width, int droplets, int radius) = 0;
};
