
#version 330 core

in vec3 vertColour;
in float depth;

out vec4 fragColour;

void main()
{
    const float FADE = 5000;
    const float FOG = 0.8;

    float eDepth = depth * depth;
    float clipped = (eDepth/FADE) > 1 ? 1 : (eDepth/FADE);

    vec4 colour = vec4(vertColour, 1) * ( 1 - clipped );
    vec4 fog = vec4(0.3f, 0.7f, 0.9f, 1.0f) * ( clipped );
    fragColour = colour + fog;
}
