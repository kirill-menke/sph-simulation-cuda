#define _USE_MATH_DEFINES
#include <cmath>
#include "visualizer.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
const float nearViewDistance = 0.1;
const float farViewDistance = 1000.0;

// camera
// Camera camera(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 0.0f, 1.0f));
Camera camera(glm::vec3(-5.5f, 4.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

#ifdef DEBUG
/* https://learnopengl.com/In-Practice/Debugging */
void APIENTRY glDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam) {
    static int errorCount = 0;

    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;

    // ignore non-significant error/warning codes
    if(id == 131169 || id == 131185 || id == 131218 || id == 131204) return; 
    std::cout << "---------------" << std::endl;
    std::cout << "Debug message (" << id << "): " <<  message << std::endl;
    switch (source) {
        case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
        case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
        case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
    } std::cout << std::endl;
    switch (type) {
        case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break; 
        case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
        case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
        case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
        case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
        case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
    } std::cout << std::endl;
    switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
        case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
    } std::cout << std::endl;
    std::cout << std::endl;
    if (severity != GL_DEBUG_SEVERITY_NOTIFICATION) {
        errorCount++;
        std::cout << boost::stacktrace::stacktrace() << std::endl;
    }
    if (errorCount >= 3) exit(0);
}
#endif

void consoleMessage() {
    char *versionGL = (char *)(glGetString(GL_VERSION));
    std::cout << "OpenGL version: " << versionGL << std::endl;
    std::cout << "GLEW version: " << GLEW_VERSION << "." << GLEW_VERSION_MAJOR
        << "." << GLEW_VERSION_MINOR << "." << GLEW_VERSION_MICRO << std::endl;
}

// callback functions:

// error handling for glfw
// DEBUG
#ifdef DEBUG
static void error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error: %s\n", description);
}
#endif

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

// set by framebuffersize callback:
float aspectRatio;
int viewportWidth, viewportHeight;

glm::vec3 getCamPos() {
    return camera.Position;
}

// radians 0 - 2pi to -pi - pi
float denormalizeRadians(float r) {
    if (r > M_PI)        { r -= 2.0 * M_PI; }
    else if (r <= -M_PI) { r += 2.0 * M_PI; }
    //float deg = angle * (180.0 / M_PI);
    return r;
}
// radians -pi - pi to 0 - 2pi
float normalizeRadians(float r) {
    if (r < 0)
        r += 2.0 * M_PI;
    //float deg = angle * (180.0 / M_PI);
    return r;
}

// angle from point to point
float CoordsToAngle(int x1, int y1, int x2, int y2) {
    int deltaX = x2 - x1;
    int deltaY = y2 - y1;
    float rad = atan2(deltaY, deltaX);
    
    rad = normalizeRadians(rad);
    float deg = rad * (180.0 / M_PI);
    return deg;
}
// angle between point and point relative from 0,0
float getAngle(float v1x, float v1y, float v2x, float v2y) {
  float angle = atan2(v2y, v2x) - atan2(v1y, v1x);
  angle = denormalizeRadians(angle);
  return angle;
}
// angle between angle and point relative from 0,0
float getAngle2(float a, float v2x, float v2y) {
  float angle = atan2(v2y, v2x) - a;
  angle = denormalizeRadians(angle);
  return angle;
}

float distance(glm::vec3 wc1, glm::vec3 wc2) {
    return sqrt(pow(wc2.x-wc1.x, 2) + pow(wc2.y-wc1.y, 2) + pow(wc2.z-wc1.z, 2));
}

float distance2d(glm::vec3 wc1, glm::vec3 wc2) {
    return sqrt(pow(wc2.x-wc1.x, 2) + pow(wc2.y-wc1.y, 2));
}

float getAspectRatio() {
    return aspectRatio; //(float)viewportWidth / (float)viewportHeight;
}

float fovYtoX(float fov_y) {
    float fov_x  = atan(tan(fov_y/2) * getAspectRatio()) * 2;
    return fov_x;
}

float fovXtoY(float fov_x) {
    float fov_y  = atan(tan(fov_x/2) / getAspectRatio()) * 2; // same but * h/w (= * 1/aspectRatio)
    return fov_y;
}

float getFOVrad() {
    return glm::radians(camera.Zoom);
}

float getFOVvertRad() {
    return fovXtoY(getFOVrad());
}

// checks if point is in viewing angle
bool inFOV(float x, float y) {
    // if relative angle from view is smaller than fov/2 on each side (abs) + a little bit (10 degrees) on each side to compensate for actual size of chunks not being points
    return abs(getAngle2(-denormalizeRadians(glm::radians(camera.Yaw-90)), x-getCamPos().x, y-getCamPos().y)) <= (getFOVrad()/2.0 + glm::radians(10.0));
}

Visualizer::Visualizer(int objectNum, int maxNumTriangles, float radius, float minBoundX, float minBoundY, float minBoundZ, float maxBoundX, float maxBoundY, float maxBoundZ) {
    // DEBUG
    #ifdef DEBUG
        glfwSetErrorCallback(error_callback);
    #endif
    // init and set window creation
    // glfw: initialize and configure

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    // define used and required gl version as 3.3 and set core profile (not default compat Profile)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // statement to fix compilation on OS X
    #endif

    // DEBUG
    #ifdef DEBUG
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);  
    #endif

    //glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    // MSAA
    glfwWindowHint(GLFW_SAMPLES, 4);
    // create window (glfw window creation)

    /* Create a window and its OpenGL context */
    // gather monitor info
    const GLFWvidmode* return_struct;
    int videoModeCount; // count of supported video modes
    int count;
    GLFWmonitor** monitors = glfwGetMonitors(&count);
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    return_struct = glfwGetVideoModes(monitor, &videoModeCount); // get monitor info
    // get max resolution
    int screenheight = return_struct[videoModeCount-1].height;
    int screenwidth = return_struct[videoModeCount-1].width;
    //std::cout << screenwidth << " " << screenheight << std::endl;
    window = glfwCreateWindow(1024, 780, "Hello World", NULL, NULL);          // window
    //GLFWwindow* window = glfwCreateWindow(screenwidth, screenheight, "Hello World", monitor, nullptr);   // fullscreen with max resolution
    //window = glfwCreateWindow(1600, 900, "Hello World", glfwGetPrimaryMonitor(), nullptr);   // fullscreen with min res
    if (window == nullptr) { // !window
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    // set window as current context
    glfwMakeContextCurrent(window); // Make the window's context current
    // register callback functions
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // set viewport on window resize
    // mouse callback functions for camera control
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    // keyboard callback functions
    glfwSetKeyCallback(window, key_callback);

    glfwGetFramebufferSize(window, &viewportWidth, &viewportHeight); // init viewport
    std::cout << viewportWidth <<  " " << viewportHeight << std::endl;
    framebuffer_size_callback(window, viewportWidth, viewportHeight);// and aspectRatio

    // Transparency
    glEnable(GL_BLEND); // order doesnt matter
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // TODO test
    int test = glfwExtensionSupported("WGL_EXT_swap_control_tear");
    int test0 = glfwExtensionSupported("GLX_EXT_swap_control_tear");
    std::cout << "WGL_EXT_swap_control_tear support: " << test << " GLX_EXT_swap_control_tear: " << test0 << std::endl;

    // vsync
    glfwSwapInterval(0);

    // face culling
    ENABLE_FACE_CULLING = false;
    //glEnable(GL_CULL_FACE);
    //glCullFace(GL_FRONT); // default back
    //glFrontFace(GL_CCW); // default counter-clock-wise

    // tell GLFW to capture the mouse
    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // get open gl functions
    // Loader Init comes here after makecontextcurrent window
    if (glewInit() != GLEW_OK) {
        std::cout << "GLEW ERROR!" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // configure global opengl state
    glEnable(GL_DEPTH_TEST); // TODO for 3d?
    // MSAA
    //glEnable(GL_MULTISAMPLE); // Enabled by default on some drivers, but not all so always enable to make sure
    // both work on linux? only arb on windows?
    glEnable(GL_MULTISAMPLE_ARB);
    //glEnable(GL_POLYGON_SMOOTH); // doesnt work but looks funny

    // version Info
    std::cout << glGetString(GL_VERSION) << std::endl;
    std::cout << glGetString(GL_VENDOR) << "\n" << glGetString(GL_RENDERER) << std::endl;
    consoleMessage();

    // text init
    if (!gltInit()) {
		fprintf(stderr, "Failed to initialize glText\n");
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

    glGenBuffers(1, &vertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, vertexArray);
    glBufferData(GL_ARRAY_BUFFER, objectNum*3*sizeof(float), NULL, GL_DYNAMIC_COPY);

    glGenBuffers(1, &triangleArray);
    glBindBuffer(GL_ARRAY_BUFFER, triangleArray);
    glBufferData(GL_ARRAY_BUFFER, maxNumTriangles*9*sizeof(float), NULL, GL_DYNAMIC_COPY);

    renderer = new Renderer();
    // Shader stuff
    shader = new Shader("src/Shaders/shader");
    // Load meshes
    sphere = new Sphere(radius, 8, 8);
    box = new Box(glm::vec3(minBoundX - radius, minBoundY - radius, minBoundZ - radius), 
        glm::vec3(maxBoundX + radius, maxBoundY + radius, maxBoundZ + radius));
}

void Visualizer::draw(int objectNum) {
    // Loop until the user closes the window 
    //while (!glfwWindowShouldClose(window)) {
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    processInput(window);

    // clear
    renderer->clear(0.3f, 0.4f, 0.4f, 1.0f);
    // bind the right shader for drawing
    shader->bind();

    // draw
    shader->setFloat("near", nearViewDistance);
    shader->setFloat("far", farViewDistance);
    shader->setVec3("camPos", getCamPos());
    shader->setFloat("alpha", 1.0f);

    // pass projection matrix to shader (note that in this case it could change every frame)
    glm::mat4 projection = glm::perspective(getFOVvertRad(), getAspectRatio(), nearViewDistance, farViewDistance); // fovy (vertical); last two params: min and max view range
    shader->setMat4("projection", projection);
    // camera/view transformation
    glm::mat4 view = camera.GetViewMatrix();
    shader->setMat4("view", view);
    // calculate the model matrix for each object and pass it to shader before drawing
    glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
    model = glm::translate(model, glm::vec3( 0.0f,  0.0f,  0.0f)); // cube position
    float angle = 0;//20.0f * i;
    model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
    shader->setMat4("model", model);

    //global light source can be either positional => perspecitve OR directional => orthographic
    glm::vec3 lightPos = glm::vec3(50.0f, 50.0f, 500.0f);
    glm::vec3 lightDir = glm::normalize(glm::vec3(1, 0.5, -1));
    bool perspectiveLight = false;
    if (perspectiveLight) {
        lightDir = getCamPos() - lightPos;
    } else { // orthographicLight
        lightPos = getCamPos() + lightDir;
    }
    shader->setVec3("lightPos", lightPos);
    shader->setVec3("lightDir", lightDir);

    //get position
    glm::vec3 worldCoord = getCamPos();

    float time0 = glfwGetTime();
    float compTime = (time0-currentFrame)*1000;
    if (ENABLE_FACE_CULLING) {
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
    }

    // draw spheres
    sphere->bind();
    glBindBuffer(GL_ARRAY_BUFFER, vertexArray);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glVertexAttribDivisor(3, 1);  

    IndexBuffer* ibo = sphere->ibo;
    ibo->bind();
    glDrawElementsInstanced(GL_TRIANGLES, ibo->getCount(), GL_UNSIGNED_INT, (void*)0, objectNum);

    // draw box
    shader->setFloat("alpha", 0.2f);
    box->bind();
    box->draw(renderer);
    shader->setFloat("alpha", 1.0f);

    if (ENABLE_FACE_CULLING) {
        glDisable(GL_CULL_FACE);
    }
    float time1 = glfwGetTime();
    float drawTime = (time1-time0)*1000;

    GLTtext *text1 = gltCreateText();
    GLTtext *text2 = gltCreateText();
    GLTtext *text3 = gltCreateText();
    gltSetText(text1, "Hello World!");
    char str[200];

    static float swapTime = 0;
    static float swapTimeonly = 0;
    std::string printString = "compute: " + std::to_string(compTime) + " draw: " + std::to_string(drawTime) + " swap: " + std::to_string(swapTime) 
        + " swaponly: " + std::to_string(swapTimeonly);
    gltSetText(text3, printString.c_str());

    // HUD text
    gltBeginDraw();
    gltColor(1.0f, 1.0f, 1.0f, 0.5f);
    gltDrawText2D(text3, 0.0, 30.0, 2.0);
    gltDrawText2D(text1, 0.0f, 0.0f, 2.0f); // x=0.0, y=0.0, scale=1.0
    snprintf(str, 200, "Frame Time: %.4f FPS: %.4f camAngle: %f, %f fov: %f", (deltaTime)*1000.0, 1.0/(deltaTime), camera.Yaw, camera.Pitch, glm::degrees(getFOVrad()));
    gltSetText(text2, str);
    gltDrawText2DAligned(text2, 0.0f, (GLfloat)viewportHeight, 2.0f, GLT_LEFT, GLT_BOTTOM);
    gltEndDraw();

    gltDeleteText(text1);
    gltDeleteText(text2);
    gltDeleteText(text3);

    // swap and poll
    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
    float swapTime0 = glfwGetTime();
    glfwSwapBuffers(window); // Swap front and back buffers
    float swapTime1 = glfwGetTime();
    swapTimeonly = (swapTime1-swapTime0)*1000;
    glfwPollEvents(); // Poll for and process events
    float time2 = glfwGetTime();
    swapTime = (time2-time1)*1000;
}

void Visualizer::drawTriangles(int numTriangles) {
    // Loop until the user closes the window 
    //while (!glfwWindowShouldClose(window)) {
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    processInput(window);

    // clear
    renderer->clear(0.3f, 0.4f, 0.4f, 1.0f);
    // bind the right shader for drawing
    shader->bind();

    // draw
    shader->setFloat("near", nearViewDistance);
    shader->setFloat("far", farViewDistance);
    shader->setVec3("camPos", getCamPos());
    shader->setFloat("alpha", 1.0f);

    // pass projection matrix to shader (note that in this case it could change every frame)
    glm::mat4 projection = glm::perspective(getFOVvertRad(), getAspectRatio(), nearViewDistance, farViewDistance); // fovy (vertical); last two params: min and max view range
    shader->setMat4("projection", projection);
    // camera/view transformation
    glm::mat4 view = camera.GetViewMatrix();
    shader->setMat4("view", view);
    // calculate the model matrix for each object and pass it to shader before drawing
    glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
    model = glm::translate(model, glm::vec3( 0.0f,  0.0f,  0.0f)); // cube position
    float angle = 0;//20.0f * i;
    model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
    shader->setMat4("model", model);

    //global light source can be either positional => perspecitve OR directional => orthographic
    glm::vec3 lightPos = glm::vec3(50.0f, 50.0f, 500.0f);
    glm::vec3 lightDir = glm::normalize(glm::vec3(1, 0.5, -1));
    bool perspectiveLight = false;
    if (perspectiveLight) {
        lightDir = getCamPos() - lightPos;
    } else { // orthographicLight
        lightPos = getCamPos() + lightDir;
    }
    shader->setVec3("lightPos", lightPos);
    shader->setVec3("lightDir", lightDir);

    //get position
    glm::vec3 worldCoord = getCamPos();

    float time0 = glfwGetTime();
    float compTime = (time0-currentFrame)*1000;
    if (ENABLE_FACE_CULLING) {
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
    }

    // draw spheres
    //sphere->bind();
    VertexBuffer* vbo = new VertexBuffer(triangleArray);
    VertexArrayObject* vao = new VertexArrayObject(numTriangles);
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
    vbo->bind();
    vao->bind();
    renderer->draw(GL_TRIANGLES, *vao, nullptr, 0, 0);

    // draw box
    shader->setFloat("alpha", 0.2f);
    box->bind();
    box->draw(renderer);
    shader->setFloat("alpha", 1.0f);

    if (ENABLE_FACE_CULLING) {
        glDisable(GL_CULL_FACE);
    }
    float time1 = glfwGetTime();
    float drawTime = (time1-time0)*1000;

    GLTtext *text1 = gltCreateText();
    GLTtext *text2 = gltCreateText();
    GLTtext *text3 = gltCreateText();
    gltSetText(text1, "Hello World!");
    char str[200];

    static float swapTime = 0;
    static float swapTimeonly = 0;
    std::string printString = "compute: " + std::to_string(compTime) + " draw: " + std::to_string(drawTime) + " swap: " + std::to_string(swapTime) 
        + " swaponly: " + std::to_string(swapTimeonly);
    gltSetText(text3, printString.c_str());

    // HUD text
    gltBeginDraw();
    gltColor(1.0f, 1.0f, 1.0f, 0.5f);
    gltDrawText2D(text3, 0.0, 30.0, 2.0);
    gltDrawText2D(text1, 0.0f, 0.0f, 2.0f); // x=0.0, y=0.0, scale=1.0
    snprintf(str, 200, "Frame Time: %.4f FPS: %.4f camAngle: %f, %f fov: %f", (deltaTime)*1000.0, 1.0/(deltaTime), camera.Yaw, camera.Pitch, glm::degrees(getFOVrad()));
    gltSetText(text2, str);
    gltDrawText2DAligned(text2, 0.0f, (GLfloat)viewportHeight, 2.0f, GLT_LEFT, GLT_BOTTOM);
    gltEndDraw();

    gltDeleteText(text1);
    gltDeleteText(text2);
    gltDeleteText(text3);

    // swap and poll
    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
    float swapTime0 = glfwGetTime();
    glfwSwapBuffers(window); // Swap front and back buffers
    float swapTime1 = glfwGetTime();
    swapTimeonly = (swapTime1-swapTime0)*1000;
    glfwPollEvents(); // Poll for and process events
    float time2 = glfwGetTime();
    swapTime = (time2-time1)*1000;
}

void Visualizer::end() {
    std::cout << "Error code: " << glGetError() << std::endl;
    // cleanup
    gltTerminate();
    //glDeleteProgram(shader); // This is done by Shader class dtor
    glfwDestroyWindow(window);
    glfwTerminate(); // glfw: terminate, clearing all previously allocated GLFW resources.
    exit(EXIT_SUCCESS);
}


// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    bool shift = false;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        shift = true;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime, shift);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime, shift);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime, shift);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime, shift);

    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        camera.ProcessKeyboard(UP, deltaTime, shift);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        camera.ProcessKeyboard(DOWN, deltaTime, shift);

    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        camera.ProcessKeyboardAsMouse(PITCH_UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        camera.ProcessKeyboardAsMouse(PITCH_DOWN, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        camera.ProcessKeyboardAsMouse(YAW_LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        camera.ProcessKeyboardAsMouse(YAW_RIGHT, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
    viewportWidth = width;
    viewportHeight = height;
    aspectRatio = (float)viewportWidth / (float)viewportHeight;
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}
