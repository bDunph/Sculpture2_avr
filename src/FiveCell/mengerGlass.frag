#version 410

#define DO_FRESNEL 1
#define DO_REFLECTION 1
#define DO_REFRACTION 1


// refractive index of some common materials:
// http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/indrf.html
#define REFRACTIVE_INDEX_OUTSIDE 1.00029
#define REFRACTIVE_INDEX_INSIDE  1.125

#define MAX_RAY_BOUNCES 2 
#define OBJECT_ABSORB_COLOUR vec3(8.0, 8.0, 3.0)
#define OBJECT_ABSORB_COLOUR_2 vec3(0.3, 9.0, 9.0)

struct Moonlight {
	vec3 direction;
	vec3 colour;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
};

struct Material {
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
	float shininess;
};

struct GroundMaterial {
	//sampler2D texture;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
	float shininess;
};

uniform Moonlight moonlight;
uniform Material material;
uniform GroundMaterial ground;

const vec4 PLANE_NORMAL = vec4(0.0, 1.0, 0.0, 0.0);
const int MAX_MARCHING_STEPS = 255;
const float MIN_DIST = 0.01;
const float MAX_DIST = 100.0;
const float EPSILON = 0.0001;
const float GAMMA = 2.2;
const float REFLECT_AMOUNT = 0.02;
const float OBJ_SIZE = 1.0;
const float NO_OBJECT = 0.0;
const float PLANE_ID =  1.0;
const float MANDEL_ID = 2.0;
const float PENUMBRA_VAL = 16.0;

float mandelDist, planeDist;

uniform mat4 MVEPMat;

uniform float sineControlVal;

uniform samplerCube skyboxTex;
//uniform sampler2D groundReflectionTex;

in vec4 nearPos;
in vec4 farPos;
in vec2 texCoordsOut;

out vec4 fragColorOut; 

//----------------------------------------------------------------------------------------
// Ground plane SDF from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
// Returns distance and object ID
//----------------------------------------------------------------------------------------

float planeSDF(vec3 pos, vec4 normal){

	return dot(pos, normal.xyz) + normal.w;

}

//----------------------------------------------------------------------------------------
// Mandelbulb SDF taken from https://www.shadertoy.com/view/tdtGRj
// Returns distance and object ID
//----------------------------------------------------------------------------------------
float mandelbulbSDF(vec3 pos) {

	float Power = 2.0;
    	float r = length(pos);
    	if(r > 1.5) return r-1.2;
    	vec3 z = pos;
    	float dr = 1.0, theta, phi;
    	    for (int i = 0; i < 3; i++) {
    	    	r = length(z);
    	    	if (r>1.5) break;
    	    	theta = acos((z.y/r) + (0.01 *  sineControlVal));
    	    	//theta = acos(z.y/r);
    	    	phi = atan(z.z,z.x);
    	    	dr =  pow( r, Power-1.0)*Power*dr + 1.0;
    	    	theta *= Power;
    	    	phi *= Power;
    	    	z = pow(r,Power)*vec3(sin(theta)*cos(phi), cos(theta), sin(phi)*sin(theta)) + pos;
    	    	//z = pow(r,Power)*vec3(sin(theta * sineControlVal)*cos(phi), sin(phi)*sin(theta), cos(theta)) + pos;
    	    	//z = pow(r,Power)*vec3(sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta)) + pos;
    	    }
    	    return 0.5*log(r)*r/dr;
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
// Distance function for the whole scene
// Returns distance and object ID
//----------------------------------------------------------------------------------------
float sceneSDF(vec3 pos){

	
	mandelDist = mandelbulbSDF(pos + vec3(0.0, -1.8, 0.0));

	planeDist = planeSDF(pos, PLANE_NORMAL);

	return min(mandelDist, planeDist);
	//return planeDist;
	
}

//----------------------------------------------------------------------------------------
// * Return the shortest distance and object ID from the eyepoint to the scene surface along
// * the marching direction. If no part of the surface is found between start and end,
// * return end.
// * 
// * eye: the eye point, acting as the origin of the ray
// * marchingDirection: the normalized direction to march in
// * start: the starting distance away from the eye
// * end: the max distance away from the ey to march before giving up
//----------------------------------------------------------------------------------------
vec2 shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start, float end) {

	float objID = 0.0;
    	float depth = start;

    	for (int i = 0; i < MAX_MARCHING_STEPS; i++) {

		vec3 pointPos = eye + depth * marchingDirection;
			
		float dist = sceneSDF(pointPos);

		
        	if (dist < EPSILON) {
			
			if(dist == mandelDist) {
				objID = MANDEL_ID;
			} else if(dist == planeDist){
				objID = PLANE_ID;
			}

			return vec2(depth, objID);
        	}

        	depth += dist;

        	if (depth >= end) {
        	    return vec2(end, NO_OBJECT);
        	}

	}

    	return vec2(end, NO_OBJECT);
}

//----------------------------------------------------------------------------------------
// Soft Shadow from https://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm 
//----------------------------------------------------------------------------------------

float softShadow(vec3 rayOrigin, vec3 rayDirection, float minDist, float maxDist, float penumbra){

	float depth = minDist;
	float ph = 1e10;

	float retValue = 1.0;

	for(int i = 0; i < 32; i++){

		vec3 point = rayOrigin + rayDirection * depth;
		float distance = mandelbulbSDF(point);	

		if(distance < EPSILON) return 0.0; 
		
		depth += distance;

		float y = distance*distance/(2.0 * ph);
		float d = sqrt(distance * distance - y * y);
		retValue = min(retValue, penumbra * d / max(0.0, depth - y));
		//retValue = min(retValue, penumbra * distance / depth);
	}

	return clamp(retValue, 0.0, 1.0);
	
}

//----------------------------------------------------------------------------------------
// Estimate mandelbulb normal
//----------------------------------------------------------------------------------------
vec3 estimateNormal(vec3 p) {
    	return normalize(vec3(
    	    sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
    	    sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
    	    sceneSDF(vec3(p.x, p.y, p.z  + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
    	));
}
//----------------------------------------------------------------------------------------

vec3 phongIllumination(vec3 k_a, vec3 k_d, vec3 k_s, float shininess, vec3 p, vec3 eye) {

	//ambient
	vec3 ambient = moonlight.ambient * moonlight.colour * k_a;

   	//diffuse
	vec3 normal = estimateNormal(p);
	vec3 lightDirection = normalize(-moonlight.direction);
	float diffAngle = max(dot(normal, lightDirection), 0.0);
	vec3 diffuse = moonlight.colour * moonlight.diffuse * diffAngle * k_d;
  
	//specular
	vec3 viewDir = normalize(eye - p);
	vec3 halfwayVector = normalize(lightDirection + viewDir);
	float specularAngle = pow(max(dot(normal, halfwayVector), 0.0), shininess);	  	
	vec3 specular = moonlight.colour * moonlight.specular * (specularAngle * k_s);

	vec3 colour = ambient + diffuse + specular;
	
	return colour;
}

//===========================================================
// To calculate the fresnel reflection amout - taken from demofox's shader - 
//	https://www.shadertoy.com/view/4tyXDR
//============================================================
float FresnelReflectAmount (float refIndOut, float refIndIn, vec3 normal, vec3 incident)
{
    #if DO_FRESNEL
        // Schlick aproximation
        float r0 = (refIndOut-refIndIn) / (refIndOut+refIndIn);
        r0 *= r0;
        float cosX = -dot(normal, incident);
        if (refIndOut > refIndIn)
        {
            float n = refIndOut/refIndIn;
            float sinT2 = n*n*(1.0-cosX*cosX);
            // Total internal reflection
            if (sinT2 > 1.0)
                return 1.0;
            cosX = sqrt(1.0-sinT2);
        }
        float x = 1.0-cosX;
        float ret = r0+(1.0-r0)*x*x*x*x*x;

        // adjust reflect multiplier for object reflectivity
        ret = (REFLECT_AMOUNT + (1.0-REFLECT_AMOUNT) * ret);
        return ret;
    #else
    	return REFLECT_AMOUNT;
    #endif
}

//============================================================



//============================================================
// Returns the colour reflected or refracted from ray cast from
// surface of an object
//============================================================
vec3 GetColourFromScene(in vec3 rayPosition, in vec3 rayDirection){

	float dist = 0.0;
	vec3 colour = vec3(0.0);
	
	vec2 surfaceDist = shortestDistanceToSurface(rayPosition, rayDirection, MIN_DIST, MAX_DIST);
	if(surfaceDist.x < MAX_DIST){
		vec3 surfacePoint = rayPosition + rayDirection * surfaceDist.x;
		
    		colour = phongIllumination(material.ambient, material.diffuse, material.specular, material.shininess, surfacePoint, rayPosition);
		//colour = rayColour(surfacePoint, rayPosition, rayDirection, MANDEL_ID);
		return colour;
	} else if(surfaceDist.x == MAX_DIST && rayDirection.y < 0.0){

		//calculate point of intersection with ground plane
		// from https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection

		vec3 groundNorm = vec3(0.0, 1.0, 0.0);
		vec3 pointOnPlane = vec3(0.0, 0.0, 0.0);
		
		float denom = dot(groundNorm, rayDirection);
		vec3 lineSeg = rayPosition - pointOnPlane;
		dist = dot(lineSeg, groundNorm) / denom;

		vec3 intersectPoint = rayPosition + rayDirection * dist;

		//vec2 texCoords = vec2(intersectPoint.x, intersectPoint.z);

		//return texture(ground.texture, texCoords).rgb;
		colour = phongIllumination(ground.ambient, ground.diffuse, ground.specular, ground.shininess, intersectPoint, rayPosition);

		return colour;
	}
	
	//else return skybox
	return texture(skyboxTex, rayDirection).rgb;
}
//============================================================

//============================================================
// Simulates total internal reflection and Beer's Law 
// code based on demofox https://www.shadertoy.com/view/4tyXDR
//============================================================
vec3 GetObjectInternalRayColour(in vec3 rayPos, in vec3 rayDirection){

	float multiplier = 1.0;
	vec3 returnVal = vec3(0.0);
	float absorbDist = 0.0;
	vec3 inNorm = vec3(0.0);
	float distance = 0.0;

	for(int i = 0; i < MAX_RAY_BOUNCES; ++i){

		vec3 rayOrigin = rayPos;

		//move ray origin along the ray direction through the whole object
		//then raymarch back from that point to find the back surface
		vec3 extendedRayPos = rayPos + rayDirection * (OBJ_SIZE * 2);

		vec2 distance = shortestDistanceToSurface(extendedRayPos, -rayDirection, MIN_DIST, MAX_DIST);

		rayPos = extendedRayPos - rayDirection * distance.x;

		inNorm = estimateNormal(rayPos); 		
		inNorm = -inNorm;

		if(distance.x < 0.0) return returnVal;

		//calculate Beer's Law absorption
		absorbDist += distance.x;
		vec3 mixedColour = mix(OBJECT_ABSORB_COLOUR, OBJECT_ABSORB_COLOUR_2, 0.0);
		vec3 absorbVal = exp(-mixedColour * absorbDist);

		//calculate how much to reflect or transmit
		float reflectMult = FresnelReflectAmount(REFRACTIVE_INDEX_INSIDE, REFRACTIVE_INDEX_OUTSIDE, inNorm, rayDirection);  
		float refractMult = 1.0 - reflectMult;
		
		//add in refraction outside of the object
		vec3 refractDirection = refract(rayDirection, inNorm, REFRACTIVE_INDEX_INSIDE / REFRACTIVE_INDEX_OUTSIDE);
		returnVal += GetColourFromScene(rayPos + refractDirection * 0.001, refractDirection) * refractMult * multiplier * absorbVal;

		//add specular highlight based on refracted ray direction
		returnVal += phongIllumination(material.ambient, material.diffuse, material.specular, material.shininess, rayPos, rayOrigin) * refractMult * multiplier * absorbVal;
		
		//follow ray down internal reflection path
		rayDirection = reflect(rayDirection, inNorm);
		
		//move the ray down the ray path a bit
		rayPos = rayPos + rayDirection * 0.001;
		
		//recursively add reflectMult amout to consecutive bounces
		multiplier *= reflectMult; 
	}
		
	return returnVal;
}
//============================================================

//---------------------------------------------------------------------------
// Check what object the point intersects and return relevant colour
//---------------------------------------------------------------------------
vec3 rayColour(vec3 pos, vec3 rayOrigin, vec3 rayDirection, float objID){

	vec3 retCol = vec3(0.0);
	vec3 K_a = vec3(0.0);
	vec3 K_d = vec3(0.0);
	vec3 K_s = vec3(0.0);
	float shininess = 0.0;

	//float planeIntersection = planeSDF(pos, PLANE_NORMAL); 
	if(objID == PLANE_ID){

		//vec3 normPos = normalize(pos);
		//K_a = texture(ground.texture, texCoordsOut).rgb;
		//K_d = texture(ground.texture, texCoordsOut).rgb;
		//K_a = texture(ground.texture, normPos.xz).rgb;
		//K_d = texture(ground.texture, normPos.xz).rgb;

		K_a = ground.ambient;
		K_d = ground.diffuse;
		K_s = ground.specular;
		shininess = ground.shininess;
		
		retCol = phongIllumination(K_a, K_d, K_s, shininess, pos, rayOrigin);
		
		//calculate shadows	
		vec3 normDir = normalize(-moonlight.direction);
		float shadowVal = softShadow(pos, normDir, MIN_DIST, MAX_DIST, PENUMBRA_VAL);
		vec3 shadow = pow(vec3(shadowVal), vec3(1.5, 1.2, 1.0));

		retCol *= shadow;
				
	} else if(objID == MANDEL_ID){

		K_a = material.ambient;
    		K_d = material.diffuse;
    		K_s = material.specular;
    		shininess = material.shininess;
		
		retCol = phongIllumination(K_a, K_d, K_s, shininess, pos, rayOrigin);

		vec3 incidentNormal = estimateNormal(pos);
		vec3 returnVal = vec3(0.0);

		//following demofox blog and shadertoy for reflection etc. https://www.shadertoy.com/view/4tyXDR and https://blog.demofox.org/2017/01/09/raytracing-reflection-refraction-fresnel-total-internal-reflection-and-beers-law/

		//calculate how much to reflect or transmit
		float reflectionScaleVal = FresnelReflectAmount(REFRACTIVE_INDEX_OUTSIDE, REFRACTIVE_INDEX_INSIDE, incidentNormal, rayDirection);	

		float refractScaleVal = 1.0 - reflectionScaleVal;

		//get reflection colour
		#if DO_REFLECTION
		vec3 reflectDirection = reflect(rayDirection, incidentNormal);
		returnVal += GetColourFromScene(pos + reflectDirection * 0.001, reflectDirection) * reflectionScaleVal;	
		#endif

		//get refraction colour
		#if DO_REFRACTION
		vec3 refractDirection = refract(rayDirection, incidentNormal, REFRACTIVE_INDEX_OUTSIDE / REFRACTIVE_INDEX_INSIDE);
		returnVal += GetObjectInternalRayColour(pos + refractDirection * 0.001, refractDirection) * refractScaleVal;		
		#endif

		retCol += returnVal;

	} else if(objID == NO_OBJECT){

		retCol = vec3(0.0, 0.0, 0.0);
	}

	return retCol;
}

void main()
{

	//************* code from https://encreative.blogspot.com/2019/05/computing-ray-origin-and-direction-from.html ************//

	vec3 rayOrigin = nearPos.xyz / nearPos.w;
	vec3 rayEnd = farPos.xyz / farPos.w;
	vec3 rayDir = rayEnd - rayOrigin;
	rayDir = normalize(rayDir);	

	//rayOrigin += vec3(0.0, -1.2, 0.0);

    	vec2 dist = shortestDistanceToSurface(rayOrigin, rayDir, MIN_DIST, MAX_DIST);
    
    	if (dist.x > MAX_DIST - EPSILON) {
        	// Didn't hit anything
        	fragColorOut = vec4(0.0, 0.0, 0.0, 0.0);
			return;
    	}

    	// The closest point on the surface to the eyepoint along the view ray
    	vec3 p = rayOrigin + dist.x * rayDir;

	vec3 colour = vec3(0.0);
	vec3 shadow = vec3(0.0);

	colour = rayColour(p, rayOrigin, rayDir, dist.y);

	//gamma correction
	vec3 fragColor = pow(colour, vec3(1.0 / GAMMA));
    	fragColorOut = vec4(fragColor, 1.0);

//-----------------------------------------------------------------------------
// To calculate depth for use with rasterized material
//-----------------------------------------------------------------------------
	vec4 pClipSpace =  MVEPMat * vec4(p, 1.0);
	vec3 pNdc = vec3(pClipSpace.x / pClipSpace.w, pClipSpace.y / pClipSpace.w, pClipSpace.z / pClipSpace.w);
	float ndcDepth = pNdc.z;
	
	float d = ((gl_DepthRange.diff * ndcDepth) + gl_DepthRange.near + gl_DepthRange.far) / 2.0; 
	gl_FragDepth = d;
}
