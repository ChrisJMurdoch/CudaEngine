
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "..\..\include\logger\log.hpp"

#define USE_VAL_LAYERS true
#define WIDTH 800
#define HEIGHT 600

const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

class VulkanApp
{
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow *window;
    VkInstance instance;

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan()
    {
        createInstance();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
        }
    }

    void cleanup()
    {
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void createInstance()
    {
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

        uint32_t glfwExtensionCount = 0;
        createInfo.ppEnabledExtensionNames = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        createInfo.enabledExtensionCount = glfwExtensionCount;

#ifndef NDEBUG
        createInfo.enabledLayerCount = 0;
#else
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
#endif

        Log::check( vkCreateInstance(&createInfo, nullptr, &instance), "vkCreateInstance" );
    }
};

int main(int argc, char *argv[])
{
    VulkanApp app;

#ifdef NDEBUG
    Log::print(Log::message, "Release mode.");
    Log::set(Log::error);
#else
    Log::print(Log::message, "Debug mode.");
    Log::set(Log::message);
#endif

    try
    {
        app.run();
    }
    catch (const std::exception &e)
    {
        Log::print(Log::error, e.what());
        return 1;
    }

    return 0;
}