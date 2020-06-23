
#pragma once

#include <map>
#include <vector>
#include <string>

std::vector<std::string> split( std::string line, char d1, char d2 );

std::map<std::string, std::string> mapFile(const char *filename);

std::vector<float> vertexFile(const char *filename);
