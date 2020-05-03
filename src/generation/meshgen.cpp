
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "..\..\include\generation\meshgen.hpp"
#include "..\..\include\logger\log.hpp"

namespace meshgen
{
    const int VERTEX_SIZE = 6;
    const int COORD_INDEX = 0;
    const int COLOUR_INDEX = 3;

    float *generateVertices(float *nodes, int width, float *out, ColourScheme cs)
    {
        // Dimensions
        int quadWidth = width-1;
        float origin = (1-width) / 2;

        // Vertices
        int nVertices = pow(quadWidth, 2) * VERTEX_SIZE;

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
                    setVertex( &out[(quad*6 + 0)*VERTEX_SIZE], origin + col,   nodes[index],         origin + row,   flat, triY, cs );
                    setVertex( &out[(quad*6 + 1)*VERTEX_SIZE], origin + col+1, nodes[index+1],       origin + row,   flat, triY, cs );
                    setVertex( &out[(quad*6 + 2)*VERTEX_SIZE], origin + col,   nodes[index+width],   origin + row+1, flat, triY, cs );
                    // Tri 2
                    a = glm::vec3( (origin+col+1),   (nodes[index+1]),     (origin+row)   );
                    b = glm::vec3( (origin+col+1), (nodes[index+1+width]), (origin+row+1) );
                    c = glm::vec3( (origin+col),   (nodes[index+width]),   (origin+row+1) );
                    normal = glm::normalize( glm::cross( (a-c), (a-b) ) );
                    flat = abs(normal.y);
                    triY = (a.y + b.y + c.y) / 3;
                    setVertex( &out[(quad*6 + 3)*VERTEX_SIZE], origin + col+1, nodes[index+1],       origin + row,   flat, triY, cs );
                    setVertex( &out[(quad*6 + 4)*VERTEX_SIZE], origin + col+1, nodes[index+1+width], origin + row+1, flat, triY, cs );
                    setVertex( &out[(quad*6 + 5)*VERTEX_SIZE], origin + col,   nodes[index+width],   origin + row+1, flat, triY, cs );
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
                    setVertex( &out[(quad*6 + 0)*VERTEX_SIZE], origin + col,   nodes[index],         origin + row,   flat, triY, cs );
                    setVertex( &out[(quad*6 + 1)*VERTEX_SIZE], origin + col+1, nodes[index+1+width], origin + row+1, flat, triY, cs );
                    setVertex( &out[(quad*6 + 2)*VERTEX_SIZE], origin + col,   nodes[index+width],   origin + row+1, flat, triY, cs );
                    // Tri 2
                    a = glm::vec3( (origin+col+1),   (nodes[index+1]),     (origin+row)   );
                    b = glm::vec3( (origin+col+1), (nodes[index+1+width]), (origin+row+1) );
                    c = glm::vec3( (origin+col),   (nodes[index]),         (origin+row)   );
                    normal = glm::normalize( glm::cross( (a-c), (a-b) ) );
                    flat = abs(normal.y);
                    triY = (a.y + b.y + c.y) / 3;
                    setVertex( &out[(quad*6 + 3)*VERTEX_SIZE], origin + col+1, nodes[index+1],       origin + row,   flat, triY, cs );
                    setVertex( &out[(quad*6 + 4)*VERTEX_SIZE], origin + col+1, nodes[index+1+width], origin + row+1, flat, triY, cs );
                    setVertex( &out[(quad*6 + 5)*VERTEX_SIZE], origin + col,   nodes[index],         origin + row,   flat, triY, cs );
                }
                
                quad++;
            }
        }
        return out;
    }

    void setVertex(float *vertex, float x, float y, float z, float flat, float triY, ColourScheme cs)
    {
        // XYZ
        vertex[COORD_INDEX+0] = x;
        vertex[COORD_INDEX+1] = y;
        vertex[COORD_INDEX+2] = z;

        // RGB
        glm::vec3 snow = glm::vec3(1.0, 1.0, 1.0);
        glm::vec3 stone = glm::vec3(0.35, 0.3, 0.25);
        glm::vec3 grass = glm::vec3(0.2, 0.4, 0.0);
        glm::vec3 dirt = glm::vec3(0.35, 0.2, 0.0);
        glm::vec3 sand = glm::vec3(0.8, 0.8, 0.7);

        if (cs == water)
            setColour( vertex, flat, glm::vec3(0.1, 0.2, 0.4), glm::vec3(0.2, 0.4, 0.8), 0.7 ); // Water

        else if (triY>10)
            setColour( vertex, flat, stone, snow, 0.85, 0.9 ); // Snow

        else if (triY>5)
            setColour( vertex, flat, stone, stone, 0.85, 0.9 ); // Mountain

        else if (triY>3)
            setColour( vertex, flat, dirt, grass, 0.85, 0.9 ); // Grass

        else
            setColour( vertex, flat, dirt, sand, 0, 1.0 ); // Sand
    }

    void setColour(float *vertex, float flat, glm::vec3 steepCol, glm::vec3 flatCol, float min, float max)
    {
        // Validate input
        if ( min<0 || min>=1 || max <=0 || max >1 )
            Log::print(Log::error, "setColour: illegal args");

        // Calculate transition
        float adjusted;
        if ( min == max )
            adjusted = flat<min ? 0 : 1;
        else
            adjusted = (flat-min) / (max-min);

        // Clip transition
        adjusted = adjusted<0.0 ? 0.0 : adjusted;
        adjusted = adjusted>1.0 ? 1.0 : adjusted;

        // Set vertex
        vertex[COLOUR_INDEX+0] = ( (1-adjusted)*steepCol.r + adjusted*flatCol.r ) * flat;
        vertex[COLOUR_INDEX+1] = ( (1-adjusted)*steepCol.g + adjusted*flatCol.g ) * flat;
        vertex[COLOUR_INDEX+2] = ( (1-adjusted)*steepCol.b + adjusted*flatCol.b ) * flat;
    }
}
