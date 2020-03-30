
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "..\..\include\logger\log.hpp"

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
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

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
        pickPhysicalDevice();
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

        Log::check(vkCreateInstance(&createInfo, nullptr, &instance), "vkCreateInstance");
    }

    void pickPhysicalDevice()
    {
        // Get devices
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0)
            throw std::runtime_error("No GPU found");
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // Choose
        int highestRating = 0;
        for (const auto& device : devices) {
            int rating = rateDevice(device);
            if (rating > highestRating) {
                physicalDevice = device;
                highestRating = rating;
                break;
            }
        }

        // Validate
        if (physicalDevice == VK_NULL_HANDLE)
            throw std::runtime_error("No Vulkan GPU found");
    }

    int rateDevice(VkPhysicalDevice device)
    {
        // Get specification
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        // Log
        Log::print(Log::message, deviceProperties.deviceName, Log::NO_NEWLINE);

        // Rate device type
        int rating = 0;
        switch(deviceProperties.deviceType)
        {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                rating += 3;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                rating += 2;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                rating += 1;
                break;
        }

        Log::print( Log::message, " Rating: " + std::to_string(rating) );
        return rating;
    }
};

int main(int argc, char *argv[])
{
    VulkanApp app;

#ifdef NDEBUG
    Log::set(Log::error);
    Log::print(Log::force, "Release mode.");
#else
    Log::set(Log::message);
    Log::print(Log::force, "Debug mode.");
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

    Log::print(Log::force, "Execution complete.", Log::NO_NEWLINE);
    return 0;
}