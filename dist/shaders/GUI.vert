
#version 330 core

layout (location = 0) in vec3 vertPosition;
layout (location = 1) in vec3 vertColour;

uniform mat4 model;
uniform float time;

out vec3 colour;

void main()
{
    colour = vertColour;
    gl_Position = model * vertPosition;
}
