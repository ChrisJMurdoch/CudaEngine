
#pragma once

int initialise(GLFWwindow *&window, GLuint &programPtr);
int createWindow(GLFWwindow *&window);
int initGLAD(GLFWwindow *window);
int loadTexture(const char *filepath, GLenum rasterType, GLuint &texPtr);
void resizeCallback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);
