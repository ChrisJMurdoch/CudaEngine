
#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 colour;

out vec3 vertColour;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;

void main()
{
    vertColour = colour;
    vec4 camRelative = view * model * vec4(position, 1.0f);
    gl_Position = projection * camRelative;
}
