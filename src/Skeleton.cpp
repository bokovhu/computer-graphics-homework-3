//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2018. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kovacs Botond Janos
// Neptun : SSEGZO
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const int tessellationLevel = 20;
const float epsilon = 0.001;
const vec3 DINI_FORWARD = vec3 (0.0, 0.0, 1.0);


inline float cscf (float a) {
	return 1.0 / sinf (a);
}
inline float secf (float a) {
	return 1.0 / cosf (a);
}
inline float frand () {
	return (float) rand () / (float) RAND_MAX;
}

// Source: https://github.com/glslify/glsl-inverse/blob/master/index.glsl
inline mat4 inverse(mat4 &m) {
	float
		a00 = m.m[0][0], a01 = m.m[0][1], a02 = m.m[0][2], a03 = m.m[0][3],
		a10 = m.m[1][0], a11 = m.m[1][1], a12 = m.m[1][2], a13 = m.m[1][3],
		a20 = m.m[2][0], a21 = m.m[2][1], a22 = m.m[2][2], a23 = m.m[2][3],
		a30 = m.m[3][0], a31 = m.m[3][1], a32 = m.m[3][2], a33 = m.m[3][3],

		b00 = a00 * a11 - a01 * a10,
		b01 = a00 * a12 - a02 * a10,
		b02 = a00 * a13 - a03 * a10,
		b03 = a01 * a12 - a02 * a11,
		b04 = a01 * a13 - a03 * a11,
		b05 = a02 * a13 - a03 * a12,
		b06 = a20 * a31 - a21 * a30,
		b07 = a20 * a32 - a22 * a30,
		b08 = a20 * a33 - a23 * a30,
		b09 = a21 * a32 - a22 * a31,
		b10 = a21 * a33 - a23 * a31,
		b11 = a22 * a33 - a23 * a32,

		det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

	return mat4(
		(a11 * b11 - a12 * b10 + a13 * b09) / det,
		(a02 * b10 - a01 * b11 - a03 * b09) / det,
		(a31 * b05 - a32 * b04 + a33 * b03) / det,
		(a22 * b04 - a21 * b05 - a23 * b03) / det,
		(a12 * b08 - a10 * b11 - a13 * b07) / det,
		(a00 * b11 - a02 * b08 + a03 * b07) / det,
		(a32 * b02 - a30 * b05 - a33 * b01) / det,
		(a20 * b05 - a22 * b02 + a23 * b01) / det,
		(a10 * b10 - a11 * b08 + a13 * b06) / det,
		(a01 * b08 - a00 * b10 - a03 * b06) / det,
		(a30 * b04 - a31 * b02 + a33 * b00) / det,
		(a21 * b02 - a20 * b04 - a23 * b00) / det,
		(a11 * b07 - a10 * b09 - a12 * b06) / det,
		(a00 * b09 - a01 * b07 + a02 * b06) / det,
		(a31 * b01 - a30 * b03 - a32 * b00) / det,
		(a20 * b03 - a21 * b01 + a22 * b00) / det
    );
}

inline vec4 quaternion (vec3 axis, float angle) {
	float s = sinf (angle / 2.0);
	return vec4 (
		axis.x * s,
		axis.y * s,
		axis.z * s,
		cosf (angle / 2.0)
	);
}

inline vec4 qmul (vec4 q1, vec4 q2) {
	vec3 d1 = vec3 (q1.x, q1.y, q1.z);
	vec3 d2 = vec3 (q2.x, q2.y, q2.z);
	vec3 c = cross (d1, d2);
	return vec4 (
		d2.x * q1.w + d1.x * q2.w + c.x,
		d2.y * q1.w + d1.y * q2.w + c.y,
		d2.z * q1.w + d1.z * q2.w + c.z,
		q1.w * q2.w - dot(d1, d2)
	);
}

inline vec3 rotate (vec3 u, vec4 q) {
	vec4 qinv = (-q.x, -q.y, -q.z, q.w);
	vec4 qr = qmul (qmul (q, vec4 (u.x, u.y, u.z, 0)), qinv);
	return vec3 (qr.x, qr.y, qr.z);
}

inline vec4 vecRotation (vec3 a, vec3 b) {
	vec3 start = normalize (a);
	vec3 dest = normalize (b);
	float cosTheta = dot (start, dest);
	vec3 rotationAxis;
	if (cosTheta < -1.0 + epsilon) {
		rotationAxis = cross (vec3 (0, 0, 1), start);
		if (length (rotationAxis) < epsilon) {
			rotationAxis = cross (vec3 (1, 0, 0), start);
		}
		rotationAxis = normalize (rotationAxis);
		return quaternion (rotationAxis, M_PI);
	}
	rotationAxis = cross (start, dest);
	float s = sqrtf ( (1.0 + cosTheta) * 2.0 );
	float invS = 1.0 / s;
	return vec4 (s * 0.5, rotationAxis.x * invS, rotationAxis.y * invS, rotationAxis.z * invS);
}

inline vec4 orient (vec3 forward, vec3 dir, vec3 up) {
	vec3 right = cross (dir, up);
	vec3 realUp = cross (right, dir);
	vec4 rot1 = vecRotation (forward, dir);
	vec3 newUp = rotate (vec3 (0, 1, 0), rot1);
	vec4 rot2 = vecRotation (newUp, realUp);
	return qmul (rot2, rot1);
}

mat4 qRotationMatrix (vec4 q) {
	return mat4 (
		// First row
		q.x * q.x + q.y * q.y - q.z * q.z - q.w * q.w, 
		2.0 * q.y * q.z - 2.0 * q.x * q.w,
		2.0 * q.y * q.w + 2.0 * q.x * q.z,
		0,

		// Second row
		2.0 * q.y * q.z + 2.0 * q.x * q.w,
		q.x * q.x - q.y * q.y + q.z * q.z - q.w * q.w,
		2.0 * q.z * q.w - 2.0 * q.x * q.y,
		0,

		// Third row
		2.0 * q.y * q.w - 2.0 * q.x * q.z,
		2.0 * q.z * q.w + 2.0 * q.x * q.y,
		q.x * q.x - q.y * q.y - q.z * q.z + q.w * q.w,
		0,

		// Forth row
		0, 0, 0, 1
	);
}

struct Camera {

	vec3 position, target, up;
	float fov, aspectRatio, nearDistance, farDistance;

public:
	Camera() {
		aspectRatio = (float)windowWidth/windowHeight;
		fov = M_PI * 0.4;
		nearDistance = 0.1; 
		farDistance = 100.0;
	}
	mat4 GetViewMatrix () { // view matrix: translates the center to the origin
		vec3 w = normalize(position - target);
		vec3 u = normalize(cross(up, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(position * (-1)) * mat4(u.x, v.x, w.x, 0,
											 u.y, v.y, w.y, 0,
											 u.z, v.z, w.z, 0,
											 0,   0,   0,   1);
	}
	mat4 GetProjectionMatrix () {
		return mat4(1.0 / (tan(fov / 2.0) * aspectRatio), 0, 0, 0,
					0, 1.0 / tan(fov / 2.0), 0, 0,
					0, 0, -(nearDistance + farDistance) / (farDistance - nearDistance), -1.0,
					0, 0, -2.0 * nearDistance * farDistance / (farDistance - nearDistance), 0);
	}
	void Animate(float t) { }
};

struct Material {
	vec3 Kd, Ks, Ka;
	float shininess;

	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.kd", name);
		Kd.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ks", name);
		Ks.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ka", name);
		Ka.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.shininess", name);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform shininess cannot be set\n");
	}
};

struct Light {

	vec3 La, Le;
	vec4 position;

	void Animate(float t) {	}

	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.La", name);
		La.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.Le", name);
		Le.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.wLightPos", name);
		position.SetUniform(shaderProg, buffer);
	}
};

struct CheckerBoardTexture : public Texture {

	CheckerBoardTexture(
			const int width = 0, 
			const int height = 0, 
			const vec3 color1 = vec3 (1.0, 1.0, 0.0), 
			const vec3 color2 = vec3 (0.0, 0.0, 1.0)
	) : Texture() {

		glBindTexture(GL_TEXTURE_2D, textureId);

		std::vector<vec3> image;

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				image.push_back ( ( (x & 1) ^ (y & 1) ) ? color1 : color2 );
			}
		}

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

};

struct RenderState {

	mat4 model;
	mat4 modelInverse;
	mat4 view;
	mat4 projection;
	mat4 modelViewProjection;
	std::vector <Light*> lights;
	std::vector <Material*> materials;
	vec3 cameraPosition;
	vec3 cameraDirection;
	Material* material = nullptr;
	Texture* texture = nullptr;

};

char* uniformModel = "u_model";
char* uniformModelInverse = "u_modelInverse";
char* uniformView = "u_view";
char* uniformProjection = "u_projection";
char* uniformMVP = "u_MVP";
char* uniformLights = "u_lights";
char* uniformNumLights = "u_numLights";
char* uniformMaterial = "u_material";
char* uniformTexture = "u_texture";
char* uniformEye = "u_eye";
char* uniformEyeDir = "u_eyeDir";

class Shader : public GPUProgram {

protected:
	void SetUniforms (RenderState &state) {

		state.model.SetUniform (getId (), uniformModel);
		state.modelInverse.SetUniform (getId (), uniformModelInverse);
		state.view.SetUniform (getId (), uniformView);
		state.projection.SetUniform (getId (), uniformProjection);
		state.modelViewProjection.SetUniform (getId (), uniformMVP);
		state.model.SetUniform (getId (), uniformModel);

		int location = glGetUniformLocation (getId (), uniformNumLights);
		if (location >= 0) {
			glUniform1i (location, state.lights.size ());
		}

		char buf [255];
		for (int i = 0; i < state.lights.size (); i++) {
			auto &l = state.lights [i];

			sprintf (buf, "%s[%d].Le", uniformLights, i);
			l->Le.SetUniform (getId (), buf);

			sprintf (buf, "%s[%d].La", uniformLights, i);
			l->La.SetUniform (getId (), buf);

			sprintf (buf, "%s[%d].position", uniformLights, i);
			l->position.SetUniform (getId (), buf);

		}

		if (state.texture) {
			glActiveTexture (GL_TEXTURE0);
			glBindTexture (GL_TEXTURE_2D, state.texture->textureId);
			location = glGetUniformLocation (getId (), uniformTexture);
			if (location >= 0) {
				glUniform1i (location, GL_TEXTURE0);
			}
		}

		if (state.material) {

			sprintf (buf, "%s.Ka", uniformMaterial);
			state.material->Ka.SetUniform (getId (), buf);

			sprintf (buf, "%s.Kd", uniformMaterial);
			state.material->Kd.SetUniform (getId (), buf);

			sprintf (buf, "%s.Ks", uniformMaterial);
			state.material->Ks.SetUniform (getId (), buf);

			sprintf (buf, "%s.shininess", uniformMaterial);
			location = glGetUniformLocation (getId (), buf);
			if (location >= 0) {
				glUniform1f (location, state.material->shininess);
			}

		}

		state.cameraPosition.SetUniform (getId (), uniformEye);
		state.cameraDirection.SetUniform (getId (), uniformEyeDir);

	}

public:
	virtual void Bind(RenderState &state) = 0;
};


struct Vertex {
	vec3 position;
	vec3 normal;
	vec2 texCoord;
};

struct AABB {
	vec3 center;
	vec3 halfDimensions;
};

class Drawable {

protected:
	GLuint vao;
	GLuint vbo;

	void AddToRenderState (RenderState &state) {

		mat4 rot;
		if (useQuaternion) {
			rot = qRotationMatrix (rotationQuaternion);
		} else {
			rot = RotationMatrix (rotation, rotationAxis);
		}

		state.model = ScaleMatrix (scale)
			* rot
			* TranslateMatrix (translate);
		state.modelInverse = inverse (state.model);
		state.modelViewProjection = state.model * state.view * state.projection;
		state.texture = texture;
		state.material = material;

	}

	virtual void OnPostInit () = 0;

public:

	vec3 translate;
	vec3 scale;
	vec3 rotationAxis;
	float rotation;
	vec4 rotationQuaternion;
	bool useQuaternion = false;
	Shader* shader;
	Texture* texture;
	Material* material;
	AABB boundingBox;

	virtual void CalculateBoundingBox () = 0;

	void Init () {

		glGenVertexArrays (1, &vao);
		glBindVertexArray (vao);

		glGenBuffers (1, &vbo);
		glBindBuffer (GL_ARRAY_BUFFER, vbo);

		glEnableVertexAttribArray (0);
		glEnableVertexAttribArray (1);
		glEnableVertexAttribArray (2);

		glVertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, (3 + 3 + 2) * sizeof (float), (void*) 0);
		glVertexAttribPointer (1, 3, GL_FLOAT, GL_FALSE, (3 + 3 + 2) * sizeof (float), (void*) (3 * sizeof (float)));
		glVertexAttribPointer (2, 2, GL_FLOAT, GL_FALSE, (3 + 3 + 2) * sizeof (float), (void*) (6 * sizeof (float)));

		glBindVertexArray (0);
		glBindBuffer (GL_ARRAY_BUFFER, 0);

		OnPostInit ();

	}

	virtual void DrawGeometry (RenderState &state) = 0;

	void Draw (RenderState &state) {

		AddToRenderState (state);
		shader->Bind (state);
		DrawGeometry (state);

	}

};

class Surface : public Drawable {

protected:

	int numTriangleStrips;
	int numVerticesPerStrip;

	std::vector <Vertex> vertices;

	Vertex vertex (float u, float v) {

		float _u = transU (u);
		float _v = transV (v);

		Vertex vtx;
		vtx.position = surfPos (_u, _v);
		vtx.normal = surfNormal (_u, _v);
		vtx.texCoord = vec2 (u, v);
		return vtx;

	}

	void OnPostInit () {

		Create ();

	}

public:

	virtual float transU (float uIn) = 0;
	virtual float transV (float vIn) = 0;

	virtual vec3 surfPos (float u, float v) = 0;
	virtual vec3 surfNormal (float u, float v) = 0;

	void CalculateBoundingBox () {

		float minX = vertices [0].position.x;
		float maxX = vertices [0].position.x;
		float minY = vertices [0].position.y;
		float maxY = vertices [0].position.y;
		float minZ = vertices [0].position.z;
		float maxZ = vertices [0].position.z;

		mat4 model = ScaleMatrix (scale)
			* RotationMatrix (rotation, rotationAxis)
			* TranslateMatrix (translate);

		for (int i = 1; i < vertices.size (); i++) {

			vec4 vPos = vec4 (
				vertices [i].position.x,
				vertices [i].position.y,
				vertices [i].position.z,
				1.0
			) * model;

			if (minX > vPos.x) minX = vPos.x;
			if (maxX < vPos.x) maxX = vPos.x;
			if (minY > vPos.y) minY = vPos.y;
			if (maxY > vPos.y) maxY = vPos.y;
			if (minZ > vPos.z) minZ = vPos.z;
			if (maxZ > vPos.z) maxZ = vPos.z;
		}

		boundingBox.center = vec3 (
			(maxX + minX) / 2.0 + translate.x,
			(maxY + minY) / 2.0 + translate.y,
			(maxZ + minZ) / 2.0 + translate.z
		);
		boundingBox.halfDimensions = vec3 (
			abs (maxX - minX) / 2.0,
			abs (maxY - minY) / 2.0,
			abs (maxZ - minZ) / 2.0
		);

	}

	void Create (int uSubdivCount = 50, int vSubdivCount = 50) {

		vertices.clear ();
		numVerticesPerStrip = (vSubdivCount + 1) * 2;
		numTriangleStrips = uSubdivCount;

		for (int i = 0; i < uSubdivCount; i++) {
			for (int j = 0; j <= vSubdivCount; j++) {
				float v0 = (float) i / (float) vSubdivCount;
				float v1 = (float) (i + 1) / (float) vSubdivCount;
				float u = (float) j / (float) uSubdivCount;
				vertices.push_back (vertex (u, v0));
				vertices.push_back (vertex (u, v1));
			}
		}

		glBindBuffer (GL_ARRAY_BUFFER, vbo);
		glBufferData (
			GL_ARRAY_BUFFER,
			sizeof (float) * (3 + 3 + 2) * vertices.size (),
			&vertices [0],
			GL_STATIC_DRAW
		);
		glBindBuffer (GL_ARRAY_BUFFER, 0);

	}

	void DrawGeometry (RenderState &state) {

		glBindVertexArray (vao);
		for (int i = 0; i < numTriangleStrips; i++) {

			glDrawArrays (GL_TRIANGLE_STRIP, i * numVerticesPerStrip, numVerticesPerStrip);

		}
		glBindVertexArray (0);

	}

};

class Sphere : public Surface {

public:
	float transU (float u) {
		return u * 2.0 * M_PI;
	}
	float transV (float v) {
		return v * M_PI;
	}

	vec3 surfPos (float u, float v) {
		return vec3 (
			cosf (u) * sinf (v),
			sinf (u) * sinf (v),
			cosf (v)
		);
	}
	vec3 surfNormal (float u, float v) {
		return normalize (surfPos (u, v));
	}

};

class DiniSurface : public Surface {

public:

	float a = 1.0;
	float b = 0.15;

	float transU (float u) {
		return u * 4.0 * M_PI;
	}
	float transV (float v) {
		return 0.01 + v * (1.0 - 0.01);
	}

	vec3 surfPos (float u, float v) {
		return vec3 (
			a * cosf (u) * sinf (v),
			a * sinf (u) * sinf (v),
			a * ( cosf (v) + logf (tanf (v / 2.0)) ) + b * u
		);
	}
	vec3 surfNormal (float u, float v) {
		vec3 du, dv;

		du = vec3 (
			-a * sinf (u) * sinf (v),
			a * cosf (u) * sinf (v),
			b
		);
		dv = vec3 (
			a * cosf (u) * cosf (v),
			a * sinf (u) * cosf (v),
			0.5 * a * ( cscf (v / 2.0) * secf (v / 2.0) - 2.0 * sinf (v) )
		);

		return normalize( cross( du, dv ) ) ;
	}

};

class KleinBottle : public Surface {

public:

	float a, b, c;

	float transU (float u) {
		return u * 2.0 * M_PI;
	}
	float transV (float v) {
		return v * 2.0 * M_PI;
	}

	vec3 surfPos (float u, float v) {

		float _a = a * cosf (u) * (1.0 + sinf (u));
		float _b = b * sinf (u);
		float _c = c * (1.0 - cosf (u) / 2.0);

		float fx = u >= M_PI ? (_a + _c * cosf (v + M_PI)) : (_a + _c * cosf (u) * cosf (v));
		float fy = u >= M_PI ? (_b) : (_b + _c * sinf (u) * cosf (v));
		float fz = _c * sinf (v);

		return vec3 (fx, fy, fz);
	}
	vec3 surfNormal (float u, float v) {

		vec3 du, dv;

		if (u >= M_PI) {

			// derivative of a * cos (u) * (1 + sin (u)) + ((c * (1 - cos (u) / 2)) * cos (v + pi)) with respect to u/v
			float dXdU = -a * sinf (u) * (sinf (u) + 1.0)
				+ a * cosf (u) * cosf (u)
				+ c * (1.0 - cosf (u) / 2.0) * cosf (u) * cosf (v)
				+ 0.5 * c * sinf (u) * sinf (u) * cosf (v);
			float dXdV = 0.5 * c * sinf (u) * (cosf (u) - 2.0) * sinf (v);
			float dYdU = b * cosf (u);
			float dYdV = 0;
			float dZdU = 0.5 * c * sinf (u) * sinf (v);
			float dZdV = 0.5 * c * (cosf (u) - 2.0) * cosf (v);

			du = vec3 (dXdU, dYdU, dZdU);
			dv = vec3 (dXdV, dYdV, dZdV);

		} else {

			// derivative of a * cos (u) * (1 + sin (u)) + ( (c * (1 - cos (u) / 2)) * cos (u) * cos (v) ) with respect to u/v
			float dXdU = -sinf (u) * (a * sinf (u) + a + c * cosf (v))
				+ a * powf (cosf (u), 2.0) + c * sinf (u) * cosf (u) * cosf (v);
			float dXdV = 0.5 * c * (cosf (u) - 2.0) * cosf (u) * sinf (v);
			float dYdU = b * cosf (u) - 0.5 * c * (cosf (u) - 2.0) * cosf (u) * cosf (v)
				+ 0.5 * c * powf (sinf (u), 2.0) * cosf (v);
			float dYdV = 0.5 * c * sinf (u) * (cosf (u) - 2.0) * sinf (v);
			float dZdU = 0.5 * c * sinf (u) * sinf (v);
			float dZdV = 0.5 * c * (cosf (u) - 2.0) * cosf (v);

			du = vec3 (dXdU, dYdU, dZdU);
			dv = vec3 (dXdV, dYdV, dZdV);

		}

		return normalize( cross( du, dv ) );
	}

	vec3 surfTangent (float u, float v) {
		vec3 du;

		if (u >= M_PI) {

			// derivative of a * cos (u) * (1 + sin (u)) + ((c * (1 - cos (u) / 2)) * cos (v + pi)) with respect to u/v
			float dXdU = a * powf (cosf (u), 2.0)
				- 0.5 * sinf (u) * (2.0 * a * (sinf (u) + 1.0) + c * cosf (v));
			float dYdU = b * cosf (u);
			float dZdU = 0.5 * c * sinf (u) * sinf (v);

			du = vec3 (dXdU, dYdU, dZdU);

		} else {

			// derivative of a * cos (u) * (1 + sin (u)) + ( (c * (1 - cos (u) / 2)) * cos (u) * cos (v) ) with respect to u/v
			float dXdU = -sinf (u) * (a * sinf (u) + a + c * cosf (v))
				+ a * powf (cosf (u), 2.0) + c * sinf (u) * cosf (u) * cosf (v);
			float dYdU = b * cosf (u) - 0.5 * c * (cosf (u) - 2.0) * cosf (u) * cosf (v)
				+ 0.5 * c * powf (sinf (u), 2.0) * cosf (v);
			float dZdU = 0.5 * c * sinf (u) * sinf (v);

			du = vec3 (dXdU, dYdU, dZdU);

		}

		return normalize( du ) ;
	}

};

inline vec4 sunQuaternion (float t) {
	return vec4 (
		cosf (t / 4.0),
		sinf (t / 4.0) * cosf (t) / 2.0,
		sinf (t / 4.0) * sinf (t) / 2.0,
		sinf (t / 4.0) * sqrtf (3.0 / 4.0)
	);
}

struct Game {

	std::vector <Light*> lights;
	std::vector <Material*> materials;
	Texture* texture;

};

class PhongShader : public Shader {

	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 position;
		};

		uniform mat4 u_model;
		uniform mat4 u_modelInverse;
		uniform mat4 u_MVP;
		uniform Light[8] u_lights;    // light sources 
		uniform int   u_numLights;
		uniform vec3  u_eye;         // pos of eye

		layout(location = 0) in vec3  a_position;            // pos in modeling space
		layout(location = 1) in vec3  a_normal;      	 // normal in modeling space
		layout(location = 2) in vec2  a_uv;

		out vec3 v_worldNormal;
		out vec3 v_worldView;
		out vec3 v_worldLightDirection[8];
		out vec2 v_texCoord;

		void main() {
			gl_Position = vec4(a_position, 1) * u_MVP;
			
			vec4 worldPosition = vec4(a_position, 1) * u_model;
			for(int i = 0; i < u_numLights; i++) {
				v_worldLightDirection[i] = u_lights[i].position.xyz  * worldPosition.w 
					- worldPosition.xyz * u_lights[i].position.w;
			}
		    v_worldView  = u_eye * worldPosition.w - worldPosition.xyz;
		    v_worldNormal = (u_modelInverse * vec4(a_normal, 0)).xyz;
		    v_texCoord = a_uv;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 position;
		};

		struct Material {
			vec3 Kd, Ks, Ka;
			float shininess;
		};

		uniform Material u_material;
		uniform Light[8] u_lights;    // light sources 
		uniform int   u_numLights;
		uniform sampler2D u_texture;

		in  vec3 v_worldNormal;       // interpolated world sp normal
		in  vec3 v_worldView;         // interpolated world sp view
		in  vec3 v_worldLightDirection[8];     // interpolated world sp illum dir
		in  vec2 v_texCoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(v_worldNormal);
			vec3 V = normalize(v_worldView); 
			if (dot(N, V) < 0) N = -N;
			vec3 texColor = texture(u_texture, v_texCoord).rgb;
			vec3 ka = u_material.Ka * texColor;
			vec3 kd = u_material.Kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < u_numLights; i++) {
				vec3 L = normalize(v_worldLightDirection[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * u_lights[i].La + (kd * cost + u_material.Ks * pow(cosd, u_material.shininess)) * u_lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState &state) {
		glUseProgram(getId()); 		// make this program run
		SetUniforms (state);
	}
};

//---------------------------
class NPRShader : public Shader {
//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform	vec4  wLightPos;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal, wView, wLight;				// in world space
		out vec2 texcoord;

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight = wLightPos.xyz * wPos.w - wPos.xyz * wLightPos.w;
		   wView  = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		uniform sampler2D diffuseTexture;

		in  vec3 wNormal, wView, wLight;	// interpolated
		in  vec2 texcoord;
		out vec4 fragmentColor;    			// output goes to frame buffer

		void main() {
		   vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
		   float y = (dot(N, L) > 0.5) ? 1 : 0.5;
		   if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
		   else						 fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
		}
	)";
public:
	NPRShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState &state) {
		glUseProgram(getId()); 		// make this program run
		SetUniforms (state);
	}
};

Texture* texture;
Material* material;
Camera* camera;
// DiniSurface* sphere;
std::vector <Drawable*> drawables;
Shader* shader;
std::vector <Light*> lights;

// Initialization, create an OpenGL context
void onInitialization() {

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);


	texture = new CheckerBoardTexture (16, 16);

	material = new Material ();
	material->Kd = vec3 (0.7, 0.5, 0.9);
	material->Ks = vec3 (2.0, 1.0, 2.0);
	material->Ka = vec3 (0.4, 0.5, 0.6);
	material->shininess = 150.0;

	camera = new Camera ();
	camera->position = vec3 (0.0, 0.0, -40.0);
	camera->target = vec3 (0.0, 0.0, 0.0);
	camera->up = vec3 (0.0, 1.0, 0.0);

	KleinBottle* kb = new KleinBottle ();
	kb->a = 6.0;
	kb->b = 16.0;
	kb->c = 4.0;
	kb->Init ();
	kb->translate = vec3 (0.0, 0.0, 0.0);
	kb->rotationAxis = vec3 (0.0, 1.0, 0.0);
	kb->rotation = 0;
	kb->scale = vec3 (1.0, 1.0, 1.0);
	kb->texture = texture;
	kb->material = material;
	drawables.push_back (kb);


	for (int i = 0; i < 15; i++) {
		DiniSurface* ds = new DiniSurface ();
		ds->Init ();
		float u = kb->transU (frand ());
		float v = kb->transV (frand ());
		ds->translate = kb->surfPos (
			u, v
		);
		vec4 q = orient (
			DINI_FORWARD,
			kb->surfNormal (
			 	u, v
			) * -1.0,
			kb->surfTangent (
				u, v
			)
		);
		ds->rotationQuaternion = q;
		ds->rotationAxis = vec3 (0.0, 1.0, 0.0);
		ds->rotation = 0.0;
		ds->useQuaternion = false;
		ds->scale = vec3 (2.0, 2.0, 2.0);
		ds->texture = texture;
		ds->material = material;
		drawables.push_back (ds);
	}

	shader = new PhongShader ();
	for (auto &d : drawables) {
		d->shader = shader;
	}

	Light* light = new Light ();
	light->position = vec4 (2.0, -2.0, -4.0, 0.0);
	light->Le = vec3 (2.5, 2.5, 2.5);
	light->La = vec3 (0.5, 0.2, 0.2);
	lights.push_back (light);
	
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	
	// sphere->Draw (state);
	for (auto &d : drawables) {
		RenderState state;
		for (auto &l : lights) {
			state.lights.push_back (l);
		}
		state.view = camera->GetViewMatrix ();
		state.projection = camera->GetProjectionMatrix ();
		state.cameraPosition = camera->position;
		state.cameraDirection = normalize (camera->target - camera->position);
		d->Draw (state);
	}

	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { }

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	float time = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	camera->position = vec3 (
		sinf (time) * 40.0,
		0.0,
		cosf (time) * 40.0
	);

	glutPostRedisplay();
}
