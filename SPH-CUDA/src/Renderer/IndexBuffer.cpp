#include "IndexBuffer.h"

IndexBuffer::IndexBuffer(const unsigned int* data, int count) : count(count) {
    glGenBuffers(1, &ib_ID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib_ID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, count*sizeof(data[0]), data, GL_DYNAMIC_DRAW);
}

IndexBuffer::~IndexBuffer() {
    glDeleteBuffers(1, &ib_ID);
}

void IndexBuffer::bind() {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib_ID);
}

int IndexBuffer::getCount() {
    return count;
}

void IndexBuffer::unbind() {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}