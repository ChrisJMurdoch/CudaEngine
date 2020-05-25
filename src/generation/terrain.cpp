
#define _USE_MATH_DEFINES
#include <functional>
#include <cmath>

#include <glm/glm.hpp>

#include "..\..\include\generation\terrain.hpp"
#include "..\..\include\logger\log.hpp"

namespace terrain
{
    // NUMBER GENERATION

    template <class T>
    int centHash(T x)
    {
        static std::hash<T> hash;
        return ( hash(x) % 201 ) - 100;
    }

    template <class T>
    int combine(T x, T y) {
        return (x*12345) + y;
    }

    float lerp(float a, float b, float x)
    {
        return a + x * (b - a);
    }

    float fade(float x) {
        return x * x * x * (x * (x * 6 - 15) + 10);
    }

    // SAMPLING

    float msin(int x, int y, float period)
    {
        float xd = ( sin( x * (2*M_PI) / period ) + 1 ) / 2;
        float yd = ( sin( y * (2*M_PI) / period ) + 1 ) / 2;
        return xd * yd;
    }

    float hash(int x, int y, float period)
    {
        return centHash( combine(x, y) ) / 100.0f;
    }

    float perlin(int x, int y, float period)
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

    // GENERATION
    
    float *generateHeightMap(int width, float min, float max, float *out, float (*sampler)(int, int, float), float period, int octaves)
    {
        const float lacunarity = 2;
        const float persistance = 0.4;

        int n = pow(width, 2);

        for (int y=0; y<width; y++) for (int x=0; x<width; x++)
        {
            int i = y*width + x;

            float height = 0;
            for (int o=0; o<octaves; o++)
            {
                float ol = pow(lacunarity, o);
                float op = pow(persistance, o);
                height += ( min + ( (max-min) * sampler(x, y, period/ol) ) ) * op;

                if ( ol != 1 )
                    height -= ( min + ( (max-min) * 0.5 ) ) * op;
            }

            out[i] = height;
        }
        
        return out;
    }
}
