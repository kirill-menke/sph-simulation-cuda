#include "AttributeBufferLayout.h"

AttributeBufferLayout::AttributeBufferLayout(std::vector< std::tuple<GLenum, GLint> > layout, VertexBuffer& vb)
            : layout(layout), vb(vb), size(0) {
    for (int i = 0; i < layout.size(); i++) {
        size += std::get<1>(layout[i]) * helperGetSize(std::get<0>(layout[i]));
    }
}

GLuint helperGetSize(GLenum t) {
    GLuint s = -1;
    switch (t) {
        case GL_FLOAT : s = sizeof(GLfloat); break;

    }
    if (s < 0) std::cout << "ERROR type getSize helper function" << std::endl;
    return s;
}