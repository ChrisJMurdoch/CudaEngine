
#version 330 core

// Vertex data
layout (location = 0) in vec3 vertPosition;
layout (location = 1) in vec3 vertColour;

// Standard uniforms
uniform float time;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 colour;

void main()
{
    colour = vertColour;
    gl_Position = model * vertPosition;
}
