
#include "..\..\include\graphic\display.hpp"
#include "..\..\include\models\vModel.hpp"
#include "..\..\include\models\eModel.hpp"
#include "..\..\include\generation\heightmap.hpp"
#include "..\..\include\generation\mesh.hpp"
#include "..\..\include\math\cpuMathEngine.hpp"
#include "..\..\include\math\gpuMathEngine.hpp"
#include "..\..\include\logger\log.hpp"
#include "..\..\include\util\io.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <thread>

int main( int argc, char *argv[] )
{
    // Determine math engine
    bool hardware = argc>1 && strcmp( argv[1], "cuda" ) == 0;
    MathEngine *math = hardware ? (MathEngine *)&GPUMathEngine() : (MathEngine *)&CPUMathEngine();
    Log::print( Log::message, hardware ? "Using Cuda acceleration." : "Not using Cuda acceleration." );

    // Create OpenGL display
    Display display = Display();

	// Add shader programs
	GLuint terrainProg = display.addShader( "shaders\\Terrain.vert", "shaders\\FShader.frag" );
	GLuint waterProg = display.addShader( "shaders\\Water.vert", "shaders\\FShader.frag" );

    // Generate terrain models
	VModel terrain = VModel( Mesh( Heightmap( mapFile("assets/generation/terrain.kval"), 500, math ), Mesh::landscape ), terrainProg, glm::vec3(0,0,0), GL_STREAM_DRAW );
	VModel water = VModel( Mesh( Heightmap( mapFile("assets/generation/water.kval"), 500, math ), Mesh::water ), waterProg, glm::vec3(0,0,0), GL_STATIC_DRAW );

    // Add terrain models
    display.addModel(terrain);
    display.addModel(water);

    // Start rendering loop
    display.start();

    // Cleanup
	glDeleteProgram(terrainProg);
	glDeleteProgram(waterProg);
}
