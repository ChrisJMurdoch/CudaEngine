
#pragma once

int loadTexture(const char *filepath, GLenum rasterType, GLuint *texPtr);
int initialise(GLFWwindow **window, GLuint *programPtr);
int createWindow(GLFWwindow **window);
int initGLAD(GLFWwindow *window);
void resizeCallback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);
