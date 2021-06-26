#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtx/normal.hpp>
#include <GLFW/glfw3.h>

#include "Renderer/IndexBuffer.h"
#include "Renderer/AttributeBufferLayout.h"
#include "Renderer/VertexArrayObject.h"
#include "Renderer/VertexBuffer.h"
#include "Renderer/Renderer.h"

#include <vector>

class Box {
public:
    Box(glm::vec3 min, glm::vec3 max);
    void draw(Renderer* renderer);
    void bind();
    int count;
    int bufferLength;
    float* buffer;
    glm::vec3 min, max;
    int createBox(std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals, std::vector<unsigned int>& indices, glm::vec3 min, glm::vec3 max);
    VertexBuffer* vbo;
    VertexArrayObject* vao;
    AttributeBufferLayout* abl;
    //IndexBuffer* ibo;
};
