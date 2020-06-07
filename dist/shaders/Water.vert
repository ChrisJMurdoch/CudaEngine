
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
    // Morph wave
    vec3 wave = position;
    float a = sin( (wave.x + time)*0.3f ) * sin( (wave.z + time)*0.3f );
    float b = sin( (-wave.x + time)*0.6f ) * sin( (-wave.z + time)*0.6f );
    wave.y += a + b;

    vertColour = colour;
    vec4 camRelative = view * model * vec4(wave, 1.0f);
    gl_Position = projection * camRelative;
}
