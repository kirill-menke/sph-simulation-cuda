#pragma once

#include <GL/glew.h>
#include "AttributeBufferLayout.h"

class VertexArrayObject {
public:
    GLuint vao_ID;
    VertexArrayObject(int count);
    void addABL(AttributeBufferLayout& abl);
    void bind();
    int getCount();
    static void unbind();
private:
    int curr_idx;
    int count;
};