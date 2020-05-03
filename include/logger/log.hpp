
#pragma once

class Log
{
public:
    enum Level { debug=0, message=1, warning=2, error=3, force=4 };
    static const bool NEWLINE = true;
    static const bool NO_NEWLINE = false;

private:
    static Level level;

public:
    static void set(Level lvl);
    template <class T>
    static void print(Level lvl, T message, bool newline=true);
    static void check(int result, const char *op);
};
