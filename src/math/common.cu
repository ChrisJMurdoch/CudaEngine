
// This has to be directly included into a translation unit as it contains
// device code, wrap include statement in a namespace to avoid linker errors.

// Enums
#include "..\..\include\math\mathEngine.hpp"

__host__ __device__
float floatHash(int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return ( x % 10000 ) / 9999.0f;
}

__host__ __device__
int combine(int x, int y) {
    return (x*12345) + y;
}

__host__ __device__
float lerp(float a, float b, float x)
{
    return a + x * (b - a);
}

__host__ __device__
float diverge(float x)
{
    const float PI = 3.14159265358979323846;
    return 0.5 - ( cos(x*PI)*0.5 );
}

__host__ __device__
float squash(float x)
{
    const float PI = 3.14159265358979323846;
    return acos( -2*(x-0.5) ) / PI;
}

__host__ __device__
float falloff(float x)
{
    const float PI = 3.14159265358979323846;
    return powf( sin(x*PI), 0.05 );
}

// SAMPLES (X,Y,P) => Z

__host__ __device__
float hashSample(int x, int y, float period)
{
    return floatHash( combine(x, y) );
}

__host__ __device__
float sinSample(int x, int y, float period)
{
    const float PI = 3.14159265358979323846;
    float xd = ( sin( x * (2*PI) / period ) + 1 ) / 2;
    float yd = ( sin( y * (2*PI) / period ) + 1 ) / 2;
    return xd * yd;
}

__host__ __device__
float perlinSample(int x, int y, float period)
{
    // Square coords
    int X = std::floor( x / period );
    int Y = std::floor( y / period );

    // Relative point coords
    float rx = (x/period) - X;
    float ry = (y/period) - Y;

    // Square corner vectors
    glm::vec2 BL = glm::normalize( glm::vec2( floatHash( combine( X , Y ) )-0.5, floatHash( combine( X , Y )+1 )-0.5 ) );
    glm::vec2 BR = glm::normalize( glm::vec2( floatHash( combine(X+1, Y ) )-0.5, floatHash( combine(X+1, Y )+1 )-0.5 ) );
    glm::vec2 TL = glm::normalize( glm::vec2( floatHash( combine( X ,Y+1) )-0.5, floatHash( combine( X ,Y+1)+1 )-0.5 ) );
    glm::vec2 TR = glm::normalize( glm::vec2( floatHash( combine(X+1,Y+1) )-0.5, floatHash( combine(X+1,Y+1)+1 )-0.5 ) );

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

    // Interpolate using diverge
    float bottom = lerp( BLd, BRd, diverge(point.x) );
    float top = lerp( TLd, TRd, diverge(point.x) );
    float centre = lerp( bottom, top, diverge(point.y) );

    // 0-1
    return (centre+1) / 2;
}

__host__ __device__
float perlinRidgeSample(int x, int y, float period)
{
    float neg = ( perlinSample(x, y, period)*2 ) - 1 ;
    return 1 - abs( neg );
}

__host__ __device__
float perlinCutSample(int x, int y, float period)
{
    float neg = ( perlinSample(x, y, period)*2 ) - 1 ;
    return abs( neg );
}

// SAMPLE COMPOSITES

__host__ __device__
float fractal(int x, int y, float period, MathEngine::Sample sample, int octaves)
{
    // Octaves
    float height = 0;
    float max = 0;
    for (int o=0; o<octaves; o++)
    {
        // Caluculate amplitude and period
        const float lacunarity = 0.5, persistance = 0.4;
        float pmult = pow(lacunarity, o), amplitude = pow(persistance, o);

        // Get sample value
        switch ( sample )
        {
        case MathEngine::hash:
            height += hashSample( x, y, pmult*period ) * amplitude;
            break;
        case MathEngine::sin:
            height += sinSample( x, y, pmult*period ) * amplitude;
            break;
        case MathEngine::perlin:
            height += perlinSample( x, y, pmult*period ) * amplitude;
            break;
        case MathEngine::perlinRidge:
            height += perlinRidgeSample( x, y, pmult*period ) * amplitude;
            break;
        default:
            height += hashSample( x, y, pmult*period ) * amplitude;
            break;
        }
        max += amplitude;
    }
    return height / max;
}

__host__ __device__
float mountain(int x, int y, float period)
{
    // Amplitudes
    float plateAmp = 32;
    float mountainAmp =  16;
    float a3 =  8;
    float a4 =  4;
    float a5 =  2;
    float a6 =  1;

    // Terrain samples
    float plate    = perlinSample(x, y, period);
    float mountain = perlinSample(x, y, period/2);
    float s3       = perlinSample(x, y, period/4);
    float s4       = perlinSample(x, y, period/8);
    float s5       = perlinSample(x, y, period/16);
    float s6       = perlinSample(x, y, period/32);

    // Diverge and flatten plate
    plate = diverge(diverge(diverge(plate)));
    plate = 0.4 + (plate*0.2);

    // Diverge mountains
    mountain = diverge( mountain );

    // Smooth surface
    s3 = squash( s3 );

    // Merge
    float amp = plateAmp + mountainAmp + a3 + a4 + a5 + a6;
    float total = ( (plate*plateAmp) + (mountain*mountainAmp) + (s3*a3) + (s4*a4) + (s5*a5)  + (s6*a6) ) / amp;
    return total;
}
