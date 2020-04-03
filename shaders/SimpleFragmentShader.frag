
#version 330 core

in vec3 vertColour;
in vec2 vertTexPos;

out vec4 fragColour;

uniform sampler2D tex1;
uniform sampler2D tex2;
  
void main()
{
    fragColour = mix(texture(tex1, vertTexPos), texture(tex2, vertTexPos), 0.2);
}
