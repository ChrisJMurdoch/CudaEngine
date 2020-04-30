
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "..\..\include\models\mesh.hpp"
#include "..\..\include\logger\log.hpp"

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
                    glm::vec3 a = glm::vec3( (origin+col),   (nodes[index]),       (origin+row)   );
                    glm::vec3 b = glm::vec3( (origin+col+1), (nodes[index+1]),     (origin+row)   );
                    glm::vec3 c = glm::vec3( (origin+col),   (nodes[index+width]), (origin+row+1) );
                    glm::vec3 normal = glm::normalize( glm::cross( (a-c), (a-b) ) );
                    float flat = abs(normal.y);
                    float triY = (a.y + b.y + c.y) / 3;
                    setVertex( &vertices[(quad*6 + 0)*VERTEX_SIZE], origin + col,   nodes[index],         origin + row,   flat, triY );
                    setVertex( &vertices[(quad*6 + 1)*VERTEX_SIZE], origin + col+1, nodes[index+1],       origin + row,   flat, triY );
                    setVertex( &vertices[(quad*6 + 2)*VERTEX_SIZE], origin + col,   nodes[index+width],   origin + row+1, flat, triY );
                    // Tri 2
                    a = glm::vec3( (origin+col+1),   (nodes[index+1]),     (origin+row)   );
                    b = glm::vec3( (origin+col+1), (nodes[index+1+width]), (origin+row+1) );
                    c = glm::vec3( (origin+col),   (nodes[index+width]),   (origin+row+1) );
                    normal = glm::normalize( glm::cross( (a-c), (a-b) ) );
                    flat = abs(normal.y);
                    triY = (a.y + b.y + c.y) / 3;
                    setVertex( &vertices[(quad*6 + 3)*VERTEX_SIZE], origin + col+1, nodes[index+1],       origin + row,   flat, triY );
                    setVertex( &vertices[(quad*6 + 4)*VERTEX_SIZE], origin + col+1, nodes[index+1+width], origin + row+1, flat, triY );
                    setVertex( &vertices[(quad*6 + 5)*VERTEX_SIZE], origin + col,   nodes[index+width],   origin + row+1, flat, triY );
                }
                else
                {
                    // Tri 1
                    glm::vec3 a = glm::vec3( (origin+col),   (nodes[index]),         (origin+row)   );
                    glm::vec3 b = glm::vec3( (origin+col+1), (nodes[index+1+width]), (origin+row+1) );
                    glm::vec3 c = glm::vec3( (origin+col),   (nodes[index+width]),   (origin+row+1) );
                    glm::vec3 normal = glm::normalize( glm::cross( (a-c), (a-b) ) );
                    float flat = abs(normal.y);
                    float triY = (a.y + b.y + c.y) / 3;
                    setVertex( &vertices[(quad*6 + 0)*VERTEX_SIZE], origin + col,   nodes[index],         origin + row,   flat, triY );
                    setVertex( &vertices[(quad*6 + 1)*VERTEX_SIZE], origin + col+1, nodes[index+1+width], origin + row+1, flat, triY );
                    setVertex( &vertices[(quad*6 + 2)*VERTEX_SIZE], origin + col,   nodes[index+width],   origin + row+1, flat, triY );
                    // Tri 2
                    a = glm::vec3( (origin+col+1),   (nodes[index+1]),     (origin+row)   );
                    b = glm::vec3( (origin+col+1), (nodes[index+1+width]), (origin+row+1) );
                    c = glm::vec3( (origin+col),   (nodes[index]),         (origin+row)   );
                    normal = glm::normalize( glm::cross( (a-c), (a-b) ) );
                    flat = abs(normal.y);
                    triY = (a.y + b.y + c.y) / 3;
                    setVertex( &vertices[(quad*6 + 3)*VERTEX_SIZE], origin + col+1, nodes[index+1],       origin + row,   flat, triY );
                    setVertex( &vertices[(quad*6 + 4)*VERTEX_SIZE], origin + col+1, nodes[index+1+width], origin + row+1, flat, triY );
                    setVertex( &vertices[(quad*6 + 5)*VERTEX_SIZE], origin + col,   nodes[index],         origin + row,   flat, triY );
                }
                
                quad++;
            }
        }
        return vertices;
    }

    void setVertex(float *vertex, float x, float y, float z, float flat, float triY)
    {
        // XYZ
        vertex[COORD_INDEX+0] = x;
        vertex[COORD_INDEX+1] = y;
        vertex[COORD_INDEX+2] = z;

        // RGB
        if (triY>20)
        {
            if (flat >= 0.65)
            {
                vertex[COLOUR_INDEX+0] = 0.1 + flat*0.4 + 2*(flat-0.65);
                vertex[COLOUR_INDEX+1] = 0.1 + flat*0.4 + 2*(flat-0.65);
                vertex[COLOUR_INDEX+2] = 0.1 + flat*0.4 + 2*(flat-0.65);
            }
            else
            {
                vertex[COLOUR_INDEX+0] = 0.1 + flat*0.4;
                vertex[COLOUR_INDEX+1] = 0.1 + flat*0.4;
                vertex[COLOUR_INDEX+2] = 0.1 + flat*0.4;
            }
        }
        else if (triY>4)
        {
            if (flat >= 0.85)
            {
                vertex[COLOUR_INDEX+0] = 0.1 + flat*0.4 - 2*(flat-0.85);
                vertex[COLOUR_INDEX+1] = 0.1 + flat*0.4 + 2*(flat-0.85);
                vertex[COLOUR_INDEX+2] = 0.1 + flat*0.4 - 2*(flat-0.85);
            }
            else
            {
                vertex[COLOUR_INDEX+0] = 0.1 + flat*0.4;
                vertex[COLOUR_INDEX+1] = 0.1 + flat*0.4;
                vertex[COLOUR_INDEX+2] = 0.1 + flat*0.4;
            }
        }
        else if (triY>3)
        {
            vertex[COLOUR_INDEX+0] = 0.4 + flat*0.4;
            vertex[COLOUR_INDEX+1] = 0.4 + flat*0.4;
            vertex[COLOUR_INDEX+2] = 0.2 + flat*0.4;
        }
        else
        {
            vertex[COLOUR_INDEX+0] = 0.1 + flat*0.3;
            vertex[COLOUR_INDEX+1] = 0.2 + flat*0.3;
            vertex[COLOUR_INDEX+2] = 0.6 + flat*0.3;
        }
    }
}
