#pragma once

#include <GL/glew.h>
#include <vector>
#include <utility>
#include <tuple>
#include <iostream>

#include "VertexBuffer.h"

class AttributeBufferLayout {
public:
    AttributeBufferLayout(std::vector< std::tuple<GLenum, GLint> > layout, VertexBuffer& vb);
    GLint size;
    std::vector< std::tuple<GLenum, GLint> > layout;
    VertexBuffer& vb;
};

GLuint helperGetSize(GLenum t);