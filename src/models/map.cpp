
#include <cmath>
#include <cstdlib>

#include "..\..\include\models\map.hpp"
#include "..\..\include\logger\log.hpp"

namespace map
{
    const int VERTEX_SIZE = 6;
    const int COORD_INDEX = 0;
    const int COLOUR_INDEX = 3;

    float *mapVertices(int w, int &n)
    {
        n = pow(w, 2);
        float origin = (float)(w-1) / -2;
        float *array = new float[n*VERTEX_SIZE];

        for (int i=0; i<n; i++)
        {
            int base = i*VERTEX_SIZE;
            float r = (float)rand() / (float)RAND_MAX;
            const float hvary = 1;
            const float cvary = 0.9;

            // COORD_INDEX
            float height = hvary * (r - 0.5);
            array[base+COORD_INDEX+0] = origin + (float)(i % w);
            array[base+COORD_INDEX+1] = height;
            array[base+COORD_INDEX+2] = origin + (float)(i / w);

            // COLOUR_INDEX
            float shade = (1-cvary)/2 + (r*cvary);
            array[base+COLOUR_INDEX+0] = shade;
            array[base+COLOUR_INDEX+1] = shade;
            array[base+COLOUR_INDEX+2] = shade;
        }
        return array;
    }
    unsigned int *mapIndices(int w, int &n)
    {
        int qw = w-1;
        n = 6 * pow(qw, 2);
        unsigned int *array = new unsigned int[n];

        int index=0;
        for (int row=0; row<qw; row++)
        {
            for(int col=0; col<qw; col++)
            {
                // Quad
                int topleft = row*w + col;

                array[index++] = topleft;
                array[index++] = topleft+1;
                array[index++] = topleft+w;

                array[index++] = topleft+1;
                array[index++] = topleft+1+w;
                array[index++] = topleft+w;
            }
        }
        return array;
    }
}