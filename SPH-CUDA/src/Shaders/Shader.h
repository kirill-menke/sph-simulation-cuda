#pragma once

#include <string>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector>

class Shader {
public:
	Shader(const std::string& fileName);
    Shader(std::vector<std::string> fileNames);
    
	void bind();
	//void Update(const Transform& transform, const Camera& camera);

	virtual ~Shader();

	// utility uniform functions
    void setBool(const std::string &name, bool value) const {         
        glUniform1i(glGetUniformLocation(program_ID, name.c_str()), (int)value); 
    }
    void setInt(const std::string &name, int value) const { 
        glUniform1i(glGetUniformLocation(program_ID, name.c_str()), value); 
    }
    void setFloat(const std::string &name, float value) const { 
        glUniform1f(glGetUniformLocation(program_ID, name.c_str()), value); 
    }
    void setVec2(const std::string &name, const glm::vec2 &value) const { 
        glUniform2fv(glGetUniformLocation(program_ID, name.c_str()), 1, &value[0]); 
    }
    void setVec2(const std::string &name, float x, float y) const { 
        glUniform2f(glGetUniformLocation(program_ID, name.c_str()), x, y); 
    }
    void setVec3(const std::string &name, const glm::vec3 &value) const { 
        glUniform3fv(glGetUniformLocation(program_ID, name.c_str()), 1, &value[0]); 
    }
    void setVec3(const std::string &name, float x, float y, float z) const { 
        glUniform3f(glGetUniformLocation(program_ID, name.c_str()), x, y, z); 
    }
    void setVec4(const std::string &name, const glm::vec4 &value) const { 
        glUniform4fv(glGetUniformLocation(program_ID, name.c_str()), 1, &value[0]); 
    }
    void setVec4(const std::string &name, float x, float y, float z, float w) const { 
        glUniform4f(glGetUniformLocation(program_ID, name.c_str()), x, y, z, w); 
    }
    void setMat2(const std::string &name, const glm::mat2 &mat) const {
        glUniformMatrix2fv(glGetUniformLocation(program_ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
    }
    void setMat3(const std::string &name, const glm::mat3 &mat) const {
        glUniformMatrix3fv(glGetUniformLocation(program_ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
    }
    void setMat4(const std::string &name, const glm::mat4 &mat) const {
        glUniformMatrix4fv(glGetUniformLocation(program_ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
    }

protected:
private:
	unsigned int NUM_SHADERS;
    GLuint shader_types[3] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_GEOMETRY_SHADER};
	unsigned int NUM_UNIFORMS = 3;
	//void operator=(const Shader& shader) {}
	//Shader(const Shader& shader) {}

	std::string loadShader(const std::string& fileName);
	void checkShaderError(GLuint shader, GLuint flag, bool isProgram, const std::string& errorMessage);
	GLuint createShader(const std::string& text, unsigned int type);

	GLuint program_ID;
	std::vector<GLuint> shaders;
	std::vector<GLuint> m_uniforms;
};