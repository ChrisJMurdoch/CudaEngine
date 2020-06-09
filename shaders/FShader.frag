
#version 330 core

in vec3 colour;
in vec3 worldPosition;

uniform vec3 focus;

out vec4 fragColour;

const float glowRadius = 10;
const float glowWidth = 2;
const vec4 glowColour = vec4( 0.0f, 0.5f, 1.0f, 1.0f );

void main()
{
    // Calculate glow
    float x = length( vec2(focus.x, focus.z) - vec2(worldPosition.x, worldPosition.z) );
    float glow = 1 - pow( ( (x-glowRadius) * 2 * (1/glowWidth)), 2 );
    glow = glow<0 ? 0 : glow;

    fragColour = glow*glowColour + vec4( colour, 1 );
}
