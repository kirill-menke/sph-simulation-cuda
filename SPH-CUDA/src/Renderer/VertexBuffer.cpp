#include "VertexBuffer.h"

VertexBuffer::VertexBuffer(const void* data, unsigned int size) {
    glGenBuffers(1, &vb_ID);
    glBindBuffer(GL_ARRAY_BUFFER, vb_ID);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);
}

VertexBuffer::VertexBuffer(GLuint vertexBuffer_ID) : vb_ID(vertexBuffer_ID) {
    glBindBuffer(GL_ARRAY_BUFFER, vb_ID);
}

VertexBuffer::~VertexBuffer() {
    glDeleteBuffers(1, &vb_ID);
}

void VertexBuffer::bind() {
    glBindBuffer(GL_ARRAY_BUFFER, vb_ID);
}

void VertexBuffer::unbind() {
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}