

#pragma once

class Log
{
public:
    enum Level { debug=0, message=1, warning=2, error=3 };
    static void set(Level lvl);
    static void print(Level lvl, const char *message);
    static void check(int result, const char *op);
private:
    static Level level;
};
