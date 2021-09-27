#define _USE_MATH_DEFINES
#include <cmath>
#include "sphere.h"

Sphere::Sphere(float radius, int rings, int sectors) : radius(radius), rings(rings), sectors(sectors) {
    std::vector<glm::vec3> vertices;
    std::vector<unsigned int> indices;
    count = createSphere(vertices, indices, radius, rings, sectors);
    bufferLength = count * 9;
    buffer = new float[bufferLength];
    int bufferInd = 0;
    for (auto &v : vertices) {
        buffer[bufferInd++] = v.x;
        buffer[bufferInd++] = v.y;
        buffer[bufferInd++] = v.z;

        // rgb color
        buffer[bufferInd++] = 0.2;
        buffer[bufferInd++] = 0.2;
        buffer[bufferInd++] = 0.8;

        buffer[bufferInd++] = -v.x;
        buffer[bufferInd++] = -v.y;
        buffer[bufferInd++] = -v.z;
    }
    vbo = new VertexBuffer(buffer, bufferLength*sizeof(buffer[0]));
    vao = new VertexArrayObject(count);
    ibo = new IndexBuffer(indices.data(), indices.size());
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

void Sphere::draw(Renderer* renderer) {
    renderer->draw(GL_TRIANGLES, *vao, ibo, 0, 0);
}

void Sphere::bind() {
    vbo->bind();
    vao->bind();
    ibo->bind();
}

inline void Sphere::push_indices(std::vector<unsigned int>& indices, int sectors, int r, int s) {
    int curRow = r * sectors;
    int nextRow = (r+1) * sectors;
    int nextS = (s+1) % sectors;

    indices.push_back(curRow + s);
    indices.push_back(nextRow + s);
    indices.push_back(nextRow + nextS);

    indices.push_back(curRow + s);
    indices.push_back(nextRow + nextS);
    indices.push_back(curRow + nextS);
}

int Sphere::createSphere(std::vector<glm::vec3>& vertices, std::vector<unsigned int>& indices, /*std::vector<vec2>& texcoords,*/
                  float radius, unsigned int rings, unsigned int sectors)
{
    int count = 0;
    float const R = 1./(float)(rings-1);
    float const S = 1./(float)(sectors-1);

    for(int r = 0; r < rings; ++r) {
        for(int s = 0; s < sectors; ++s) {
            float const y = sin( -M_PI_2 + M_PI * r * R );
            float const x = cos(2*M_PI * s * S) * sin( M_PI * r * R );
            float const z = sin(2*M_PI * s * S) * sin( M_PI * r * R );

            //texcoords.push_back(vec2(s*S, r*R));
            vertices.push_back(glm::vec3(x,y,z) * radius);
            if(r < rings-1)
                push_indices(indices, sectors, r, s);
                count += 2;
        }
    }
    return count;
}