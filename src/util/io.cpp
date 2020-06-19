
#include <util/io.hpp>

#include <logger/log.hpp>

#include <fstream>

void divide(std::string input, char delimiter, std::string &a, std::string &b)
{
    // Get character array
    const char *array = input.c_str();

    // Get delimiter location
    int i = 0, d = -1;
    while ( array[i++] != '\0' )
    {
        if ( array[i] == delimiter )
        {
            d = i;
            break;
        }
    }
    
    if ( d==-1 )
        throw "Delimiter not found";

    // Create substrings
    a = input.substr (0, d);
    b = input.substr (d+1, input.length()-d-1);
}

std::map<std::string, std::string> mapFile(const char *filename)
{
    // Open file
    std::ifstream file (filename);
    if ( !file.is_open() )
        throw "Error opening file";
    
    // Parse file into map
    std::string line;
    static std::map<std::string, std::string> map;
    while ( getline (file, line) )
    {
        try
        {
            std::string key, val;
            divide( line, ':', key, val );
            map[key] = val;
        }
        catch ( char const* e ) {}
    }
    file.close();

    return map;
}
