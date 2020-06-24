

// === ENGINE INCLUDES ===

// OpenGL interface
#include <graphic/display.hpp>
#include <graphic/instance.hpp>
#include <graphic/model.hpp>

// Procedural generation classes
#include <generation/heightmap.hpp>
#include <generation/mesh.hpp>

// Math engines
#include <math/cpuMathEngine.hpp>
#include <math/gpuMathEngine.hpp>

// Engine logging system
#include <logger/log.hpp>

// .kval parser
#include <util/io.hpp>


// === LIBRARY INCLUDES ===

// OpenGL loader
#include <glad/glad.h>

// Cuda & GLSL compatible vector library
#include <glm/glm.hpp>


// === STANDARD INCLUDES ===

#include <chrono>
#include <vector>
#include <string>


// === FUNCTIONS ===

int main( int argc, char *argv[] )
{
    // Determine math engine to use
    bool hardware = argc>1 && strcmp( argv[1], "cuda" ) == 0;
    MathEngine *math = hardware ? (MathEngine *)&GPUMathEngine() : (MathEngine *)&CPUMathEngine();
    Log::print( Log::message, hardware ? "Using Cuda acceleration." : "Not using Cuda acceleration." );

    // Profile load time
    std::chrono::steady_clock::time_point loadStart = std::chrono::steady_clock::now();

    // Create OpenGL display
    Display display = Display();

	// Add shader programs to display
	GLuint terrainProg = display.addShaderProg( "shaders/Terrain.vert", "shaders/FShader.frag" );
	GLuint waterProg = display.addShaderProg( "shaders/Water.vert", "shaders/FShader.frag" );

    // Get map width
    int width = argc>2 ? std::stoi( argv[2] ) : 500;

    // Generate water model
	Model water = Model( Mesh( Heightmap( mapFile("assets/generation/water.kval"), width, math ), Mesh::water ), waterProg, GL_STATIC_DRAW );

    // Generate terrain model
    Heightmap terrainMap( mapFile("assets/generation/terrain.kval"), width, math );
	Model terrain = Model( Mesh( terrainMap, Mesh::landscape ), terrainProg, GL_STREAM_DRAW );

    // Add instances to display
    Instance ti = Instance( &terrain, glm::vec3(0,0,0) );
    Instance wi = Instance( &water, glm::vec3(0,0,0) );
    display.addInstance(&ti);
    display.addInstance(&wi);

    // Profile load time
    std::chrono::steady_clock::time_point loadEnd = std::chrono::steady_clock::now();
    float loadTime = std::chrono::duration_cast<std::chrono::microseconds>(loadEnd - loadStart).count() / 1000000.0f;
    Log::print( Log::message, "Loading time: ", Log::NO_NEWLINE );
    Log::print( Log::message, loadTime );

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
