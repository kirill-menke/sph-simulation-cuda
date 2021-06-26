#include "Renderer.h"
#include "IndexBuffer.h"

void Renderer::draw(GLenum mode, VertexArrayObject& vao, IndexBuffer* ib, int count, int start) {
    vao.bind();
    
    if (ib == nullptr) {
        if (count == 0) count = vao.getCount();
        glDrawArrays(mode, start, count);
    } else {
        ib->bind();
        if (count == 0) count = ib->getCount();
        glDrawElements(mode, count, GL_UNSIGNED_INT, (const void *) (start * sizeof(unsigned int)));
    }
}

void Renderer::clear(float r, float g, float b, float a) {
    glClearColor(r, g, b, a); // set clear color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}