
// GLM
#include <glm/glm.hpp>

namespace testcommon
{
    #include "..\..\src\math\common.cu"
}

#include "..\..\include\math\test.hpp"
#include "..\..\include\logger\log.hpp"

void testSampling( int dimension ) {
    
    // Distribution array
    int dist[11];
    for (int i=0; i<11; i++) { dist[i]=0; }

    // Sample
    float hHash=0, lHash=1;
    for (int y=0; y<dimension; y++) for (int x=0; x<dimension; x++)
    {
        // Get sample
        float sample = testcommon::perlinSample(x, y, 10);

        // Calc range
        hHash = sample>hHash ? sample : hHash;
        lHash = sample<lHash ? sample : lHash;

        // Calc dist
        int i = (int)(sample*10);
        dist[i]++;
    }

    // Print range
    Log::print( Log::force, lHash, Log::NO_NEWLINE );
    Log::print( Log::force, "-", Log::NO_NEWLINE );
    Log::print( Log::force, hHash, Log::NEWLINE );

    // Print dist
    for(int i=0; i<11; i++)
    {
        Log::print( Log::force, dist[i], Log::NO_NEWLINE );
        Log::print( Log::force, " ", Log::NO_NEWLINE );
    }
}
