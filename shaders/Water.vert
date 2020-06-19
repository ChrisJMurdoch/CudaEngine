
#version 330 core

// Vertex data
layout (location = 0) in vec3 vertPosition;
layout (location = 1) in vec3 vertColour;

// Standard uniforms
uniform float time;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec4 colour;
out vec3 worldPosition;

void main()
{
    // World position
    vec4 wp4 = model * vec4(vertPosition, 1.0f);
    worldPosition.x = wp4.x;
    worldPosition.y = wp4.y;
    worldPosition.z = wp4.z;

    // Colour
    colour = colour = vec4( vertColour, 0.8f );

    // Morph wave
    vec3 morphed = vertPosition;
    morphed.y += ( sin( (morphed.x + time)*0.3f ) + sin( (morphed.z + time)*0.3f ) + sin( (-morphed.x + time)*0.4f ) * sin( (-morphed.z + time)*0.4f ) ) / 3;

    // Transform position
    gl_Position = projection * view * model * vec4(morphed, 1.0f);
}
