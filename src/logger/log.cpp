
#include <iostream>

#include "..\..\include\logger\log.hpp"

// Default logger to messages and up
Log::Level Log::level = Log::message;

void Log::set(Level lvl)
{
    level = lvl;
}

void Log::print(Level lvl, const char *message)
{
    if (lvl >= level)
        std::cout << message << std::endl;
}

void Log::check(int result, const char *op)
{
    if (result != 0)
        std::cout << "Op: " << op << " failed, error code: " << result << std::endl;
}
