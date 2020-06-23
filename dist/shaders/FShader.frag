
#version 330 core

in vec4 colour;
in vec3 worldPosition;
in float flatness;

uniform vec3 focus;

out vec4 fragColour;

const vec4 glowColour = vec4( 0.0f, 0.75f, 0.25f, 1.0f );

void main()
{
    // Calculate glow
    float dist = length( vec2(focus.x, focus.z) - vec2(worldPosition.x, worldPosition.z) );

    const float glowRadius = 5;
    const float glowWidth = 1;

    float glow = 1 - pow( ( (dist-glowRadius) * 2 * (1/glowWidth)), 2 );
    glow = glow<0 ? 0 : glow;

    vec4 lit = ( colour * pow(flatness, 1) );
    lit.a = colour.a;
    fragColour = lit + glow*glowColour;
}
