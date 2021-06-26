#include "Shader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

Shader::Shader(const std::string& fileName) : Shader({fileName + ".vs", fileName + ".fs",}) {
}

Shader::Shader(std::vector<std::string> fileNames) {
    NUM_SHADERS = fileNames.size();
    shaders = std::vector<GLuint>(NUM_SHADERS);
#ifdef SIMPLE_AND_FAST
    GLuint vertex_shader, fragment_shader, program;
    vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, LoadShader(fileName + ".vs").c_str, NULL);
    glCompileShader(vertex_shader);
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, LoadShader(fileName + ".fs").c_str, NULL);
    glCompileShader(fragment_shader);
    program_ID = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
#else
	program_ID = glCreateProgram();
    for(unsigned int i = 0; i < NUM_SHADERS; i++)
	    shaders[i] = createShader(loadShader(fileNames[i]), shader_types[i]);

	for(unsigned int i = 0; i < NUM_SHADERS; i++)
		glAttachShader(program_ID, shaders[i]);

	glLinkProgram(program_ID);
	checkShaderError(program_ID, GL_LINK_STATUS, true, "Error linking shader program");

	glValidateProgram(program_ID);
	checkShaderError(program_ID, GL_LINK_STATUS, true, "Invalid shader program");
    
    // delete the shaders as they are linked into our program now and no longer necessery
	for(unsigned int i = 0; i < NUM_SHADERS; i++)
        glDeleteShader(shaders[i]);
#endif
}

Shader::~Shader()
{
	glDeleteProgram(program_ID);
}

// use
void Shader::bind()
{
	glUseProgram(program_ID);
}

std::string Shader::loadShader(const std::string& fileName) {
    // retrieve the source code from source (filePath)
    std::string shaderCode;
    std::ifstream shaderFile;
    // ensure ifstream object can throw exceptions:
    shaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);
    try {
        // open file
        shaderFile.open(fileName);
        std::stringstream shaderStream;
        // read file's buffer content into stream
        shaderStream << shaderFile.rdbuf();		
        // close file handler
        shaderFile.close();
        // convert stream into string
        shaderCode = shaderStream.str();		
    }
    catch (std::ifstream::failure e) {
        std::cout << "ERROR SHADER: FILE_NOT_SUCCESFULLY_READ" << std::endl;
    }
    return shaderCode;
}

GLuint Shader::createShader(const std::string& shader_code, unsigned int type) {
    GLuint shader_ID = glCreateShader(type);
    if (shader_ID == 0) {
        std::string shaderType;
        if (type == GL_VERTEX_SHADER) {
            shaderType = "GL_VERTEX_SHADER";
        } else if (type == GL_FRAGMENT_SHADER) {
            shaderType = "GL_FRAGMENT_SHADER";
        } else {
            shaderType = "UNKNOWN";
        }
        std::cerr << "Error compiling shader type " << type << std::endl;
    }
    const GLchar* shaderCode = shader_code.c_str();
    glShaderSource(shader_ID, 1, &shaderCode, NULL);
    glCompileShader(shader_ID);
    checkShaderError(shader_ID, GL_COMPILE_STATUS, false, "Error compiling shader!");
    return shader_ID;
}

// utility function for checking shader compilation/linking errors.
void Shader::checkShaderError(GLuint shader, GLuint flag, bool isProgram, const std::string& errorMessage) {
    // flag can either be GL_COMPILE_STATUS or GL_LINK_STATUS
    GLint success = 0;
    GLchar error[1024] = { 0 };

    if(isProgram)
        glGetProgramiv(shader, flag, &success);
    else
        glGetShaderiv(shader, flag, &success);

    if(success == GL_FALSE) {
        if(isProgram)
            glGetProgramInfoLog(shader, sizeof(error), NULL, error);
        else
            glGetShaderInfoLog(shader, sizeof(error), NULL, error);

        std::string flagString;
        if (flag == GL_COMPILE_STATUS) {
            flagString = "GL_COMPILE_STATUS";
        } else if (flag == GL_LINK_STATUS) {
            flagString = "GL_LINK_STATUS";
        } else {
            flagString = "UNKNOWN";
        }
        std::cerr << errorMessage << " ERROR of type: " << flagString << std::endl << error << std::endl;
    }
}