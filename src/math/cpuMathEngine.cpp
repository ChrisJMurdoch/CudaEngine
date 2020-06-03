
#include <functional>
#include <iostream>
#include <cmath>

// GLM vector maths
#include <glm/glm.hpp>

#include "..\..\include\math\cpuMathEngine.hpp"

#define M_PI 3.14159265358979323846

// DEVICE SETUP

CPUMathEngine::CPUMathEngine() {}

// HELPER FUNCTIONS

int combine(int x, int y) {
    return (x*12345) + y;
}

float hashFunction(int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return ( x % 101 ) / 100.0f;
}

float msin(int x, int y, float period)
{
    float xd = ( sin( x * (2*M_PI) / period ) + 1 ) / 2;
    float yd = ( sin( y * (2*M_PI) / period ) + 1 ) / 2;
    return xd * yd;
}

int centHash(int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return ( x % 201 ) - 100;
}

float lerp(float a, float b, float x)
{
    return a + x * (b - a);
}

float fade(float x)
{
    return x * x * x * (x * (x * 6 - 15) + 10);
}

float falloff(float x)
{
    return pow( sin(x*M_PI), 0.05 );
}

float perlinSample(int x, int y, float period)
{
    // Square index
    int X = std::floor( x / period );
    int Y = std::floor( y / period );

    // Normal relative position
    float rx = (x/period) - X;
    float ry = (y/period) - Y;

    // Square corner vectors
    glm::vec2 BL = glm::normalize( glm::vec2( centHash( combine( X , Y ) ), centHash( combine( X , Y )+1 ) ) );
    glm::vec2 BR = glm::normalize( glm::vec2( centHash( combine(X+1, Y ) ), centHash( combine(X+1, Y )+1 ) ) );
    glm::vec2 TL = glm::normalize( glm::vec2( centHash( combine( X ,Y+1) ), centHash( combine( X ,Y+1)+1 ) ) );
    glm::vec2 TR = glm::normalize( glm::vec2( centHash( combine(X+1,Y+1) ), centHash( combine(X+1,Y+1)+1 ) ) );

    // Relational vectors
    glm::vec2 point = glm::vec2( rx, ry );
    glm::vec2 BLr = glm::vec2( 0, 0 ) - point;
    glm::vec2 BRr = glm::vec2( 1, 0 ) - point;
    glm::vec2 TLr = glm::vec2( 0, 1 ) - point;
    glm::vec2 TRr = glm::vec2( 1, 1 ) - point;

    // Dot products
    float BLd = glm::dot( BL, BLr );
    float BRd = glm::dot( BR, BRr );
    float TLd = glm::dot( TL, TLr );
    float TRd = glm::dot( TR, TRr );

    // Interpolate
    float bottom = lerp( BLd, BRd, fade(point.x) );
    float top = lerp( TLd, TRd, fade(point.x) );
    float centre = lerp( bottom, top, fade(point.y) );

    // (-1), 1  =>  0, 1
    return (centre+1) / 2;
}

// FUNCTIONS

void CPUMathEngine::generateHeightMap(int dimension, float min, float max, float *out, Sample sample, float period, int octaves)
{

    const float lacunarity = 2;
    const float persistance = 0.4;

    int n = pow(dimension, 2);

    for (int y=0; y<dimension; y++) for (int x=0; x<dimension; x++)
    {
        int i = y*dimension + x;

        float height = 0;
        for (int o=0; o<octaves; o++)
        {
            float ol = pow(lacunarity, o);
            float op = pow(persistance, o);

            switch ( sample )
            {
            case hash:
                height += ( min + ( (max-min) * hashFunction( combine(x,y) ) ) ) * op;
                break;
            case sin:
                height += ( min + ( (max-min) * msin( x, y, period ) ) ) * op;
                break;
            case perlin:
                height += ( min + ( (max-min) * perlinSample( x, y, period ) ) ) * op;
                break;
            default:
                height += ( min + ( (max-min) * hashFunction( combine(x,y) ) ) ) * op;
                break;
            }

            if ( ol != 1 )
                height -= ( min + ( (max-min) * 0.5 ) ) * op;

            // Dropoff
            float unitX = (float)x / dimension, unitY = (float)y / dimension;
            height *= falloff(unitX) * falloff(unitY);
        }

        out[i] = height;
    }
}
