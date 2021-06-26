#pragma once

#include <GL/glew.h>

class IndexBuffer {
private:
    GLuint ib_ID;
    int count;
public:
    IndexBuffer(const unsigned int* data, int count);
    ~IndexBuffer();
    void bind();
    int getCount();
    static void unbind();
};