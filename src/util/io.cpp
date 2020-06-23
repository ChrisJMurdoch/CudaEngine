
#include <util/io.hpp>

#include <logger/log.hpp>

#include <fstream>

std::vector<std::string> split( std::string line, char d1, char d2 )
{
    // Get character array
    const char *array = line.c_str();

    static std::vector<std::string> words;
    words.clear();

    // Iterate through characters
    int i = 0;
    char c = array[i++];
    bool done = false;
    while ( !done )
    {
        std::vector<char> word;

        while ( c != d1 && c != d2 && c != '\0' )
        {
            
            word.push_back(c);
            c = array[i++];
        }
        word.push_back('\0');

        if ( word.size() > 1 )
        {
            words.push_back( std::string( &word[0] ) );
        }

        if ( c == '\0' )
        {
            done = true;
            continue;
        }

        c = array[i++];
    }

    return words;
}

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

std::vector<float> vertexFile(const char *filename)
{
    // Open file
    std::ifstream file (filename);
    if ( !file.is_open() )
        throw "Error opening file";
    
    // Parse file into map
    std::string line;
    std::vector<float> floats;
    while ( getline (file, line) )
    {
        // Skip comments and empty lines
        if ( line.size() == 0 || line.at(0) == '/' )
            continue;

        std::vector<std::string> words = split( line, ',', ' ' );
        for (std::string s : words)
        {
            floats.push_back( stof( s ) );
        }
    }
    file.close();

    return floats;
}
