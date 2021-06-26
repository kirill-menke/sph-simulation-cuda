#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in vec3 aInstancePos;

out vec3 ourColor;
out vec3 Normal;
//flat out vec3 Normal;
out vec3 FragPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	gl_Position = projection * view * model * vec4(aPos + aInstancePos, 1.0f);
	ourColor = aColor;
	Normal = aNormal;
	FragPos = vec3(model * vec4(aPos, 1.0f));
}
