
#include <generation/mesh.hpp>

#include <logger/log.hpp>
#include <graphic/model.hpp>

#include <glm/glm.hpp>

float range( float min, float max, float val )
{
    if (val<min) val = min;
    if (val>max) val = max;
    return (val - min) / (max - min);
}

glm::vec3 lerp( glm::vec3 a, glm::vec3 b, float x )
{
    return x*b + (1-x)*a;
}

void setVec( float *vertex, glm::vec3 vec )
{
    vertex[0] = vec.x;
    vertex[1] = vec.y;
    vertex[2] = vec.z;
}

void setVertex(float *vertex, glm::vec3 pos, glm::vec3 normal, float triY, Mesh::ColourScheme cs)
{
    setVec( &vertex[Model::ATTR_COORDS], pos );
    setVec( &vertex[Model::ATTR_NORMAL], normal );

    const glm::vec3 waterColour = glm::vec3(0.15, 0.25, 0.4);
    const glm::vec3 sand = glm::vec3(0.8, 0.8, 0.7);
    const glm::vec3 dirt = glm::vec3(0.35, 0.2, 0.0);
    const glm::vec3 grass = glm::vec3(0.25, 0.4, 0.05);
    const glm::vec3 stone = glm::vec3(0.35, 0.3, 0.25);
    const glm::vec3 snow = glm::vec3(0.95, 0.95, 0.95);

    glm::vec3 plains = lerp( dirt, grass, range( 0.7, 0.9, normal.y ) );
    glm::vec3 mountain = lerp( stone, snow, range( 0.7, 0.9, normal.y ) );

    float SAND_DIRT = 1.5, DIRT_PLAINS = 2, PLAINS = 3, PLAINS_STONE = 5, STONE = 6, STONE_MOUNTAIN = 12, MOUNTAIN = 15;

    if ( cs == Mesh::water )
        setVec( &vertex[Model::ATTR_COLOUR], waterColour );

    else if ( triY<SAND_DIRT ) // Sand
        setVec( &vertex[Model::ATTR_COLOUR], sand );

    else if ( triY<DIRT_PLAINS ) // Sand - Dirt
        setVec( &vertex[Model::ATTR_COLOUR], lerp( sand, dirt, range( SAND_DIRT, DIRT_PLAINS, triY ) ) );

    else if ( triY<PLAINS ) // Dirt - Plains
        setVec( &vertex[Model::ATTR_COLOUR], lerp( dirt, plains, range( DIRT_PLAINS, PLAINS, triY ) ) );

    else if ( triY<PLAINS_STONE ) // Plains
        setVec( &vertex[Model::ATTR_COLOUR], plains );

    else if ( triY<STONE ) // Plains - Stone
        setVec( &vertex[Model::ATTR_COLOUR], lerp( plains, stone, range( PLAINS_STONE, STONE, triY ) ) );

    else if ( triY<STONE_MOUNTAIN ) // Stone
        setVec( &vertex[Model::ATTR_COLOUR], stone );

    else // Stone - Mountain
        setVec( &vertex[Model::ATTR_COLOUR], lerp( stone, mountain, range( STONE_MOUNTAIN, MOUNTAIN, triY ) ) );

}

void setTri( float *vertexData, glm::vec3 a, glm::vec3 b, glm::vec3 c, Mesh::ColourScheme cs )
{
    glm::vec3 normal = glm::normalize( glm::cross( (a-b), (a-c) ) );
    float flatness = normal.y;
    float triY = (a.y + b.y + c.y) / 3;
    setVertex( &vertexData[0*Model::VERTEX_STRIDE], a, normal, triY, cs );
    setVertex( &vertexData[1*Model::VERTEX_STRIDE], b, normal, triY, cs );
    setVertex( &vertexData[2*Model::VERTEX_STRIDE], c, normal, triY, cs );
}

void setQuad( float x, float y, float *primary, float *vertexData, int width, Mesh::ColourScheme cs, bool rotate )
{
    // Quad corners
    glm::vec3 a = glm::vec3( (x),   (primary[0]),       (y)   );
    glm::vec3 b = glm::vec3( (x+1), (primary[1]),       (y)   );
    glm::vec3 c = glm::vec3( (x),   (primary[width]),   (y+1) );
    glm::vec3 d = glm::vec3( (x+1), (primary[width+1]), (y+1) );

    if (rotate)
    {
        setTri( &vertexData[0*Model::VERTEX_STRIDE], a, c, b, cs );
        setTri( &vertexData[3*Model::VERTEX_STRIDE], b, c, d, cs );
    }
    else
    {
        setTri( &vertexData[0*Model::VERTEX_STRIDE], c, d, a, cs );
        setTri( &vertexData[3*Model::VERTEX_STRIDE], d, b, a, cs );
    }
}

Mesh::Mesh( float *vertexData, int nVertices )
{
    this->vertexData = vertexData;
    this->nVertices = nVertices;
}

Mesh::Mesh( Heightmap &primary, ColourScheme cs ) : Mesh( primary, primary, cs ) {}
Mesh::Mesh( Heightmap &primary, Heightmap &compare, ColourScheme cs )
{
    // Calculate dimensions
    int width = pow( primary.nNodes, 0.5 );
    int quadWidth = width-1;
    this->nVertices = pow( quadWidth, 2 ) * 6;
    this->vertexData = new float[nVertices * Model::VERTEX_STRIDE];

    // Centre model
    float offset = (1-width) / 2.0f;

    int quad = 0;
    bool toggle = false;
    for (int row=0; row<quadWidth; row++) for(int col=0; col<width; col++)
            if ( col != quadWidth ) // Skip if col & col+1 are on different rows
                setQuad( col+offset, row+offset, &primary.nodes[row*width + col], &vertexData[quad++*6*Model::VERTEX_STRIDE], width, cs, toggle=!toggle );
}

Mesh::~Mesh()
{
    delete[] vertexData;
}
