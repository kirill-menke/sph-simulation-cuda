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

class Sphere {
public:
    Sphere(int radius, int rings, int sectors);
    void draw(Renderer &renderer);
    void bind();
    int count;
    int bufferLength;
    float* buffer;
    int radius;
    int rings;
    int sectors;
    inline void push_indices(std::vector<unsigned int>& indices, int sectors, int r, int s);
    int createSphere(std::vector<glm::vec3>& vertices, std::vector<unsigned int>& indices, float radius, unsigned int rings, unsigned int sectors);
    VertexBuffer* vbo;
    VertexArrayObject* vao;
    AttributeBufferLayout* abl;
    IndexBuffer* ibo;
};
