#pragma once

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <iostream>

#include "src/Shaders/Shader.h"
#include "libs/camera.h"

#define GLT_IMPLEMENTATION
#include "libs/gltext.h"

#include <iostream>

#define BOOST_STACKTRACE_USE_ADDR2LINE
// #include <boost/stacktrace.hpp>

#include "src/Renderer/Renderer.h"
#include "src/Renderer/VertexBuffer.h"
#include "src/Renderer/IndexBuffer.h"
#include "src/Renderer/VertexArrayObject.h"
#include "src/Renderer/AttributeBufferLayout.h"

#include "src/sphere.h"
#include "src/box.h"

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/normal.hpp>
//#include <glm/gtx/string_cast.hpp>

#include <vector>

class Visualizer {
public:
    Visualizer(float radius, float minBoundX, float minBoundY, float minBoundZ, float maxBoundX, float maxBoundY, float maxBoundZ);
    void draw(float* translations, int objectNum);
    void end();
    
    GLFWwindow* window;

    Renderer* renderer;

    Shader* shader;
    
    Sphere* sphere;
    Box* box;
    bool ENABLE_FACE_CULLING;
};