#include "VertexArrayObject.h"
#include <GL/glew.h>
#include <cassert>

VertexArrayObject::VertexArrayObject(int count) : count(count), curr_idx(0) {
    glGenVertexArrays(1, &vao_ID);
}

void VertexArrayObject::addABL(AttributeBufferLayout& abl) {
    glBindVertexArray(vao_ID);

    abl.vb.bind();
    int curr_size = 0;
    for (int i = 0; i < abl.layout.size(); i++) {
        GLenum type = std::get<0>(abl.layout[i]);
        int typeSize = helperGetSize(type);
        int typeCount = std::get<1>(abl.layout[i]);
        glVertexAttribPointer(curr_idx, abl.layout.size(), type, GL_FALSE, abl.size, reinterpret_cast<void *>(curr_size));
        glEnableVertexAttribArray(curr_idx);
        curr_idx++;
        curr_size += typeSize * typeCount;
    }
    assert(curr_size == abl.size);
}

void VertexArrayObject::bind() {
    glBindVertexArray(vao_ID);
}

int VertexArrayObject::getCount() {
    return count;
}

// static
void VertexArrayObject::unbind() {
    glBindVertexArray(0);
}