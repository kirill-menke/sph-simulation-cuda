#include "box.h"

Box::Box(glm::vec3 min, glm::vec3 max) : min(min), max(max) {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<unsigned int> indices;
    count = createBox(vertices, normals, indices, min, max);
    bufferLength = count * 9;
    buffer = new float[bufferLength];
    int bufferInd = 0;
    for (int ind = 0; ind < indices.size(); ind++) {
        int i = indices[ind];
        buffer[bufferInd++] = vertices[i].x;
        buffer[bufferInd++] = vertices[i].y;
        buffer[bufferInd++] = vertices[i].z;

        // rgb color
        buffer[bufferInd++] = 0.4;
        buffer[bufferInd++] = 0.4;
        buffer[bufferInd++] = 0.7;

        buffer[bufferInd++] = normals[ind/6].x;
        buffer[bufferInd++] = normals[ind/6].y;
        buffer[bufferInd++] = normals[ind/6].z;
    }
    vbo = new VertexBuffer(buffer, bufferLength*sizeof(buffer[0]));
    vao = new VertexArrayObject(count);
    //ibo = new IndexBuffer(indices.data(), indices.size());
    AttributeBufferLayout *abl = new AttributeBufferLayout(
        {
        {GL_FLOAT, 3},
        {GL_FLOAT, 3},
        {GL_FLOAT, 3}
        },
        *vbo
    );
    vao->addABL(*abl);
    //vao->bind();
}

void Box::draw(Renderer* renderer) {
    renderer->draw(GL_TRIANGLES, *vao, nullptr, 0, 0);
}

void Box::bind() {
    vbo->bind();
    vao->bind();
    //ibo->bind();
}

int Box::createBox(std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals, std::vector<unsigned int>& indices, glm::vec3 min, glm::vec3 max)
{
    int count = 6*2*3; //8;
    vertices = {
        {min.x, min.y, min.z},
        {max.x, min.y, min.z},
        {min.x, max.y, min.z},
        {max.x, max.y, min.z},
        {min.x, min.y, max.z},
        {max.x, min.y, max.z},
        {min.x, max.y, max.z},
        {max.x, max.y, max.z},
    };

    indices = {
        0, 3, 1,
        0, 3, 2,
        0, 5, 1,
        0, 5, 4,
        0, 6, 2,
        0, 6, 4,
        1, 7, 3,
        1, 7, 5,
        2, 7, 6,
        2, 7, 3,
        4, 7, 5,
        4, 7, 6,
    };

    normals = {
        {0, 0, 1},
        {0, 1, 0},
        {1, 0, 0},
        {-1, 0, 0},
        {0, -1, 0},
        {0, 0, -1},
    };

    return count;
}