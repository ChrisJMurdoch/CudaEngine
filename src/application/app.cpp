

// === ENGINE INCLUDES ===

// OpenGL interface
#include "..\..\include\graphic\display.hpp"
#include "..\..\include\graphic\vModel.hpp"

// Procedural generation classes
#include "..\..\include\generation\heightmap.hpp"
#include "..\..\include\generation\mesh.hpp"

// Math engines
#include "..\..\include\math\cpuMathEngine.hpp"
#include "..\..\include\math\gpuMathEngine.hpp"

// Engine logging system
#include "..\..\include\logger\log.hpp"

// .kval parser
#include "..\..\include\util\io.hpp"


// === LIBRARY INCLUDES ===

// OpenGL loader
#include <glad/glad.h>

// Cuda & GLSL compatible vector library
#include <glm/glm.hpp>


// === STANDARD INCLUDES ===

#include <chrono>


// === FUNCTIONS ===

int main( int argc, char *argv[] )
{
    // Determine math engine to use
    bool hardware = argc>1 && strcmp( argv[1], "cuda" ) == 0;
    MathEngine *math = hardware ? (MathEngine *)&GPUMathEngine() : (MathEngine *)&CPUMathEngine();
    Log::print( Log::message, hardware ? "Using Cuda acceleration." : "Not using Cuda acceleration." );

    // Create OpenGL display
    Display display = Display();

	// Add shader programs to display
	GLuint terrainProg = display.addShaderProg( "shaders\\Terrain.vert", "shaders\\FShader.frag" );
	GLuint waterProg = display.addShaderProg( "shaders\\Water.vert", "shaders\\FShader.frag" );

    // Generate terrain models
    Heightmap terrainMap( mapFile("assets/generation/terrain.kval"), 500, math );
    Heightmap waterMap( mapFile("assets/generation/water.kval"), 500, math );
	VModel terrain = VModel( Mesh( terrainMap, Mesh::landscape ), terrainProg, GL_STREAM_DRAW );
	VModel water = VModel( Mesh( waterMap, Mesh::water ), waterProg, GL_STATIC_DRAW );

    // Add terrain models to display
    display.addModel(terrain);
    display.addModel(water);

    // Start rendering loop
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point lastTime = startTime;
	while ( !display.shouldClose() )
	{
        // Calculate time-delta
        std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
        float deltaTime = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - lastTime).count() / 1000000.0f;
        float totalTime = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTime).count() / 1000000.0f;
        lastTime = currentTime;

        // Refresh display
        display.refresh( totalTime, deltaTime );
    }

    // Cleanup
	glDeleteProgram(terrainProg);
	glDeleteProgram(waterProg);
}