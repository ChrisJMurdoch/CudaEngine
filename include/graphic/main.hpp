
#pragma once

int initialise(GLFWwindow *&window, GLuint &terrainProg, GLuint &waterProg);
int createWindow(GLFWwindow *&window);
int initGLAD(GLFWwindow *window);
void resizeCallback(GLFWwindow *window, int width, int height);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow *window, float deltaTime, glm::vec3 cameraFront);
