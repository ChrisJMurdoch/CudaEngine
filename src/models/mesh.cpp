
#include <cmath>

#include "..\..\include\models\mesh.hpp"

namespace mesh
{
    const int VERTEX_SIZE = 6;
    const int COORD_INDEX = 0;
    const int COLOUR_INDEX = 3;

    float *generateVertices(float *nodes, int width)
    {
        // Dimensions
        int quadWidth = width-1;
        float origin = (1-width) / 2;

        // Vertices
        int nVertices = pow(quadWidth, 2) * VERTEX_SIZE;
        float *vertices = new float[nVertices*VERTEX_SIZE];

        int quad = 0;
        bool toggle = false;
        for (int row=0; row<quadWidth; row++)
        {
            for(int col=0; col<width; col++)
            {
                // Quad
                if (col == quadWidth) // Skip if i & i+1 are straddling new row
                    continue;

                // Node index
                int index = row*width + col;

                // Switch between topleft tri and topright tri
                if (toggle = !toggle)
                {
                    // Tri 1
                    setVertex( &vertices[(quad*6 + 0)*VERTEX_SIZE], origin + col,   nodes[index],         origin + row,   1 );
                    setVertex( &vertices[(quad*6 + 1)*VERTEX_SIZE], origin + col+1, nodes[index+1],       origin + row,   1 );
                    setVertex( &vertices[(quad*6 + 2)*VERTEX_SIZE], origin + col,   nodes[index+width],   origin + row+1, 1 );
                    // Tri 2
                    setVertex( &vertices[(quad*6 + 3)*VERTEX_SIZE], origin + col+1, nodes[index+1],       origin + row,   1 );
                    setVertex( &vertices[(quad*6 + 4)*VERTEX_SIZE], origin + col+1, nodes[index+1+width], origin + row+1, 1 );
                    setVertex( &vertices[(quad*6 + 5)*VERTEX_SIZE], origin + col,   nodes[index+width],   origin + row+1, 1 );
                }
                else
                {
                    // Tri 1
                    setVertex( &vertices[(quad*6 + 0)*VERTEX_SIZE], origin + col,   nodes[index],         origin + row,   1 );
                    setVertex( &vertices[(quad*6 + 1)*VERTEX_SIZE], origin + col+1, nodes[index+1+width], origin + row+1, 1 );
                    setVertex( &vertices[(quad*6 + 2)*VERTEX_SIZE], origin + col,   nodes[index+width],   origin + row+1, 1 );
                    // Tri 2
                    setVertex( &vertices[(quad*6 + 3)*VERTEX_SIZE], origin + col+1, nodes[index+1],       origin + row,   1 );
                    setVertex( &vertices[(quad*6 + 4)*VERTEX_SIZE], origin + col+1, nodes[index+1+width], origin + row+1, 1 );
                    setVertex( &vertices[(quad*6 + 5)*VERTEX_SIZE], origin + col,   nodes[index],         origin + row,   1 );
                }
                
                quad++;
            }
        }
        return vertices;
    }

    void setVertex(float *vertex, float x, float y, float z, float flat)
    {
        // XYZ
        vertex[COORD_INDEX+0] = x;
        vertex[COORD_INDEX+1] = y;
        vertex[COORD_INDEX+2] = z;
        // RGB
        float shade = (y + 3) / 6;
        vertex[COLOUR_INDEX+0] = shade;
        vertex[COLOUR_INDEX+1] = shade;
        vertex[COLOUR_INDEX+2] = shade;
    }
}
