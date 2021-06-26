#pragma once

#include "VertexArrayObject.h"
#include "IndexBuffer.h"
#include "../Shaders/Shader.h"

class Renderer {
public:
    void draw(GLenum mode, VertexArrayObject& vao, IndexBuffer* ib, int count = 0, int start = 0);
    void clear(float r, float g, float b, float a);
};