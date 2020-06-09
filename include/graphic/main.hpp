
#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

void createWindow(GLFWwindow *&window);
void initGLAD(GLFWwindow *window);
void resizeCallback(GLFWwindow *window, int width, int height);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void scrollCallback(GLFWwindow* window, double xOff, double yOff);
void processInput(GLFWwindow *window, float deltaTime, glm::vec3 cameraFront);
