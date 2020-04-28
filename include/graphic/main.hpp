
#pragma once

int initialise(GLFWwindow *&window, GLuint &programPtr);
int createWindow(GLFWwindow *&window);
int initGLAD(GLFWwindow *window);
void resizeCallback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);
