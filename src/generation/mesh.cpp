
#include <generation\mesh.hpp>

#include <logger\log.hpp>

#include <glm/glm.hpp>

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
    vertex[Mesh::COLOUR_INDEX+0] = ( (1-adjusted)*steepCol.r + adjusted*flatCol.r ) * flat;
    vertex[Mesh::COLOUR_INDEX+1] = ( (1-adjusted)*steepCol.g + adjusted*flatCol.g ) * flat;
    vertex[Mesh::COLOUR_INDEX+2] = ( (1-adjusted)*steepCol.b + adjusted*flatCol.b ) * flat;
}

void setVertex(float *vertex, float x, float y, float sy, float z, float flat, float triY, Mesh::ColourScheme cs)
{
    // XYZ
    vertex[Mesh::COORD_INDEX+0] = x;
    vertex[Mesh::COORD_INDEX+1] = y;
    vertex[Mesh::COORD_INDEX+2] = z;

    // RGB
    const glm::vec3 stone = glm::vec3(0.35, 0.3, 0.25);
    const glm::vec3 plains = glm::vec3(0.25, 0.4, 0.05);
    const glm::vec3 dirt = glm::vec3(0.35, 0.2, 0.0);
    const glm::vec3 sediment = glm::vec3(0.5, 0.5, 0.4);
    const glm::vec3 sand = glm::vec3(0.8, 0.8, 0.7);
    const glm::vec3 snow = glm::vec3(0.95, 0.95, 0.95);
    const glm::vec3 waterC = glm::vec3(0.15, 0.15, 0.4);

    float eroded =  sy - y;

    if (cs == Mesh::water) // Water
        setColour( vertex, flat,
            glm::vec3(0.15, 0.15, 0.4),
            glm::vec3(0.3, 0.4, 0.8),
            0.7, 1.0
        );

    else if (triY>11) // Snowy
        setColour( vertex, flat, stone, snow, 0.9, 0.95 );

    else if (triY>5) // Mountain
        setColour( vertex, flat, stone, stone, 0.8, 0.9 );

    else if (triY>2) // Plains
        setColour( vertex, flat, dirt, plains, 0.8, 0.9 );

    else // Sand
        setColour( vertex, flat, sand, sand, 0.8, 0.9 );
}

Mesh::Mesh( Heightmap &primary, ColourScheme cs ) : Mesh( primary, primary, cs ) {}

Mesh::Mesh( Heightmap &primary, Heightmap &compare, ColourScheme cs )
{
    // Get data
    int width = pow( primary.nNodes, 0.5 );
    nVertices = primary.nNodes * 6;
    vertexData = new float[nVertices * VERTEX_SIZE];

    // Dimensions
    int quadWidth = width-1;
    float origin = (1-width) / 2;

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
                glm::vec3 a = glm::vec3( (origin+col),   (primary.nodes[index]),       (origin+row)   );
                glm::vec3 b = glm::vec3( (origin+col+1), (primary.nodes[index+1]),     (origin+row)   );
                glm::vec3 c = glm::vec3( (origin+col),   (primary.nodes[index+width]), (origin+row+1) );
                glm::vec3 normal = glm::normalize( glm::cross( (a-c), (a-b) ) );
                float flat = abs(normal.y);
                float triY = (a.y + b.y + c.y) / 3;
                setVertex( &vertexData[(quad*6 + 0)*VERTEX_SIZE], origin + col,   primary.nodes[index],         compare.nodes[index], origin + row,   flat, triY, cs );
                setVertex( &vertexData[(quad*6 + 1)*VERTEX_SIZE], origin + col+1, primary.nodes[index+1],       compare.nodes[index+1], origin + row,   flat, triY, cs );
                setVertex( &vertexData[(quad*6 + 2)*VERTEX_SIZE], origin + col,   primary.nodes[index+width],   compare.nodes[index+width], origin + row+1, flat, triY, cs );
                // Tri 2
                a = glm::vec3( (origin+col+1),   (primary.nodes[index+1]),     (origin+row)   );
                b = glm::vec3( (origin+col+1), (primary.nodes[index+1+width]), (origin+row+1) );
                c = glm::vec3( (origin+col),   (primary.nodes[index+width]),   (origin+row+1) );
                normal = glm::normalize( glm::cross( (a-c), (a-b) ) );
                flat = abs(normal.y);
                triY = (a.y + b.y + c.y) / 3;
                setVertex( &vertexData[(quad*6 + 3)*VERTEX_SIZE], origin + col+1, primary.nodes[index+1],       compare.nodes[index+1], origin + row,   flat, triY, cs );
                setVertex( &vertexData[(quad*6 + 4)*VERTEX_SIZE], origin + col+1, primary.nodes[index+1+width], compare.nodes[index+1+width], origin + row+1, flat, triY, cs );
                setVertex( &vertexData[(quad*6 + 5)*VERTEX_SIZE], origin + col,   primary.nodes[index+width],   compare.nodes[index+width], origin + row+1, flat, triY, cs );
            }
            else
            {
                // Tri 1
                glm::vec3 a = glm::vec3( (origin+col),   (primary.nodes[index]),         (origin+row)   );
                glm::vec3 b = glm::vec3( (origin+col+1), (primary.nodes[index+1+width]), (origin+row+1) );
                glm::vec3 c = glm::vec3( (origin+col),   (primary.nodes[index+width]),   (origin+row+1) );
                glm::vec3 normal = glm::normalize( glm::cross( (a-c), (a-b) ) );
                float flat = abs(normal.y);
                float triY = (a.y + b.y + c.y) / 3;
                setVertex( &vertexData[(quad*6 + 0)*VERTEX_SIZE], origin + col,   primary.nodes[index],         compare.nodes[index], origin + row,   flat, triY, cs );
                setVertex( &vertexData[(quad*6 + 1)*VERTEX_SIZE], origin + col+1, primary.nodes[index+1+width], compare.nodes[index+1+width], origin + row+1, flat, triY, cs );
                setVertex( &vertexData[(quad*6 + 2)*VERTEX_SIZE], origin + col,   primary.nodes[index+width],   compare.nodes[index+width], origin + row+1, flat, triY, cs );
                // Tri 2
                a = glm::vec3( (origin+col+1),   (primary.nodes[index+1]),     (origin+row)   );
                b = glm::vec3( (origin+col+1), (primary.nodes[index+1+width]), (origin+row+1) );
                c = glm::vec3( (origin+col),   (primary.nodes[index]),         (origin+row)   );
                normal = glm::normalize( glm::cross( (a-c), (a-b) ) );
                flat = abs(normal.y);
                triY = (a.y + b.y + c.y) / 3;
                setVertex( &vertexData[(quad*6 + 3)*VERTEX_SIZE], origin + col+1, primary.nodes[index+1],       compare.nodes[index+1], origin + row,   flat, triY, cs );
                setVertex( &vertexData[(quad*6 + 4)*VERTEX_SIZE], origin + col+1, primary.nodes[index+1+width], compare.nodes[index+1+width], origin + row+1, flat, triY, cs );
                setVertex( &vertexData[(quad*6 + 5)*VERTEX_SIZE], origin + col,   primary.nodes[index],         compare.nodes[index], origin + row,   flat, triY, cs );
            }
            quad++;
        }
    }
}

Mesh::~Mesh()
{
    delete[] vertexData;
}
