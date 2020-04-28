
#include <iostream>

#include "..\..\include\logger\log.hpp"

// Default logger to messages and up
Log::Level Log::level = Log::message;

void Log::set(Level lvl)
{
    level = lvl;
}

template <class T>
void Log::print(Level lvl, T message, bool newline)
{
    if (lvl >= level)
    {
        std::cout << message;
        if (newline)
            std::cout << std::endl;
    }
}
template void Log::print(Level, int, bool);
template void Log::print(Level, unsigned int, bool);
template void Log::print(Level, char *, bool);
template void Log::print(Level, std::string, bool);
template void Log::print(Level, const char *, bool);

void Log::check(int result, const char *op)
{
    if (result != 0)
        std::cout << "Op: " << op << " failed, error code: " << result << std::endl;
}
