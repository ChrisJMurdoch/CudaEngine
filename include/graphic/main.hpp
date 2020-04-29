
#pragma once

int initialise(GLFWwindow *&window, GLuint &programPtr);
int createWindow(GLFWwindow *&window);
int initGLAD(GLFWwindow *window);
void resizeCallback(GLFWwindow *window, int width, int height);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void createBuffers(
	float vertices[], int nVertices,
	unsigned int indices[], int nIndices,
	GLuint &VAO, GLuint &VBO, GLuint &EBO
);
void processInput(GLFWwindow *window, float deltaTime, glm::vec3 cameraFront);
