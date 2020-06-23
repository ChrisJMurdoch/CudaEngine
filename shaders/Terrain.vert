
#version 330 core

// Vertex data
layout (location = 0) in vec3 vertPosition;
layout (location = 1) in vec3 vertColour;
layout (location = 2) in vec3 vertNormal;

// Standard uniforms
uniform float time;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec4 colour;
out vec3 worldPosition;
out float flatness;

void main()
{
    // World position
    vec4 wp4 = model * vec4(vertPosition, 1.0f);
    worldPosition.x = wp4.x;
    worldPosition.y = wp4.y;
    worldPosition.z = wp4.z;

    // Normal based lighting
    flatness = pow( vertNormal.y, 0.75 );

    // Morph wave
    vec3 morphed = vertPosition;
    float morphAmount = 0;
    morphed.y += morphAmount;

    // Transform position
    gl_Position = projection * view * model * vec4(morphed, 1.0f);

    // Colour
    float underWater = vertPosition.y<0 ? -vertPosition.y : 0;
    underWater /= 10;
    underWater = underWater>1 ? 1 : underWater;
    colour = vec4( vertColour.r*(1-underWater), vertColour.g*(1-(underWater/2)), vertColour.b, 1 );

    // Display flatness
    const float opacity = 0;
    vec4 f = flatness>0.95 ? vec4( 0, 1, 0, 1 ) : vec4( 1, 0, 0, 1 );
    colour = colour*(1-opacity) + f*opacity;
}
