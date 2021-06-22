#pragma once
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <experimental/filesystem>
#include <ios>
#include "helper_math.h"

#include "helper_structs.h"

namespace fs = std::experimental::filesystem;

class FileManager {

    std::string input_base;
    std::string input_file;
    
public:

    FileManager(const std::string& input_base, const std::string& input_file) : 
        input_base(input_base), input_file(input_file) {};

    Parameters readParams() {
        std::unordered_map<std::string, std::string> params;
        std::ifstream paramStream(input_base + "/" + input_file);
        std::string param, value;

        while (paramStream >> param >> value) {
            params[param] = value;
        }
        paramStream.close();

        return Parameters(params);
    }
};

