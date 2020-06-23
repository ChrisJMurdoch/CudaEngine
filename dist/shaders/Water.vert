
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
    flatness = pow( vertNormal.y, 1.5 );

    // Morph wave
    vec3 morphed = vertPosition;
    float morphAmount = ( sin( (morphed.x + time)*0.3f ) + sin( (morphed.z + time)*0.3f ) + sin( (-morphed.x + time)*0.4f ) * sin( (-morphed.z + time)*0.4f ) ) / 3;
    morphed.y += morphAmount;

    // Transform position
    gl_Position = projection * view * model * vec4(morphed, 1.0f);

    // Colour
    colour = vec4( vertColour, 0.75 );
}
