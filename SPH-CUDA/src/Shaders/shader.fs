#version 330 core
out vec4 FragColor;

in vec3 ourColor;
in vec3 Normal;
//flat in vec3 Normal;

// note: FragPos == gl_FragCoord
in vec3 FragPos;

// Light perspective
uniform vec3 lightPos;
// or orthographic
uniform vec3 lightDir;
// in both cases -lightDir is the direction from cam to sun

uniform float near;
uniform float far;

uniform vec3 camPos;

// transparency
uniform float alpha;



//https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
float LinearizeDepth(float depth) 
{
    /*float z = depth * 2.0 - 1.0; // back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));*/
	// reduces to:
	return far * near / (far + depth * (near - far));
}

//https://stackoverflow.com/questions/21549456/how-to-implement-a-ground-fog-glsl-shader
//http://www.iquilezles.org/www/articles/fog/fog.htm
vec3 applyFogSimple( in vec3  rgb,       // original color of the pixel
               in float distance ) // camera to point distance
{
	float b = 1.0;
    float fogAmount = 1.0 - exp( -distance*b );
    vec3  fogColor  = vec3(0.5,0.6,0.7);
    return mix( rgb, fogColor, fogAmount );
}
vec3 applyFog( in vec3  rgb,      // original color of the pixel
               in float distance, // camera to point distance
               in vec3  rayDir,   // camera to point vector
               in vec3  sunDir )  // sun light direction
{
	float b = 1.0;
    float fogAmount = 1.0 - exp( -distance * b );
    float sunAmount = max( dot( rayDir, sunDir ), 0.0 );
    vec3  fogColor  = mix( vec3(0.5, 0.6, 0.7), // bluish
                           vec3(1.0, 0.9, 0.7), // yellowish
                           pow(sunAmount, 2.0) );
    return mix( rgb, fogColor, fogAmount );
}
/*
Another variation of the technique is to split the usual mix() command in its two parts, ie, replace

finalColor = mix( pixelColor, fogColor, exp(-distance*b) );


with

finalColor = pixelColor*(1.0-exp(-distance*b)) + fogColor*exp(-distance*b);


Now, according to classic CG atmospheric scattering papers, the first term could be interpreted as the absortion of light due to scattering or "extinction", and the second term can be interpreted as the "inscattering". We note that this way of expressing fog is more powerfull, because now we can choose independent fallof parameters b for the extinction and inscattering. Furthermore, we can have not one or two, but up to six different coefficients - three for the rgb channels of the extintion color and three for the rgb colored version of the inscattering.

vec3 extColor = vec3( exp(-distance*be.x), exp(-distance*be.y) exp(-distance*be.z) );
vec3 insColor = vec3( exp(-distance*bi.x), exp(-distance*bi.y) exp(-distance*bi.z) );
finalColor = pixelColor*(1.0-extColor) + fogColor*insColor;
*/

vec3 applyFogNonConst( in vec3  rgb,      // original color of the pixel
               in float distance, // camera to point distance
               in vec3  rayOri,   // camera position
               in vec3  rayDir )  // camera to point vector
{
	float c = 1.0;
	float b = 0.01;
    float fogAmount = c * exp(-rayOri.z*b) * (1.0-exp( -distance*rayDir.z*b ))/rayDir.z;
    vec3  fogColor  = vec3(0.5,0.6,0.7);
    return mix( rgb, fogColor, fogAmount );
}

void main()
{
	// light calcualtions
	vec3 norm = normalize(Normal);

	// use this for global point light (perspective)
	//vec3 lightDir = normalize(lightPos - FragPos);
	// and this for global orthogonal light (orthographic)
	vec3 lightDir = lightDir;

	float diff = max(dot(norm, lightDir), 0.0);
	vec3 diffColor = diff * ourColor;

	float dist = LinearizeDepth(gl_FragCoord.z);
	float depth = dist / far; // divide by far for demonstration
    vec3 depthColor = vec3(depth);
	//FragColor = vec4(depthColor, 1.0);

	//FragColor = vec4(mix(diffColor, depthColor, 0.5), 1.0);
	//FragColor = vec4(applyFogSimple(diffColor, depth), 1.0);
	vec3 fragDir = normalize(FragPos - camPos); 
	//FragColor = vec4(applyFog(diffColor, depth, fragDir, -lightDir), 1.0);
	//FragColor = vec4(applyFogNonConst(diffColor, dist, camPos, fragDir), alpha);
	FragColor = vec4(diffColor, alpha);
}
