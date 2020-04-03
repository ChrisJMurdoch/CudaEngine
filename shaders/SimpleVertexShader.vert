
#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 colour;
layout (location = 2) in vec2 texPos;

out vec3 vertColour;
out vec2 vertTexPos;

void main()
{
    gl_Position = vec4(position, 1.0);
    vertColour = colour;
    vertTexPos = texPos;
}
