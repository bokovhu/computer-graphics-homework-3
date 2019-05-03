
class KleinBottle : public ParamSurface {

public:
	float a, b, c;
	KleinBottle (float _a, float _b, float _c) : a (_a), b (_b), c (_c) { Create (); }

	vec3 pos (float r, float u, float v) {
		if (u >= M_PI) {
			return vec3 (
				a * cosf (u) * (1.0 + sinf (u)) - r * cosf (v + M_PI),
				b * sinf (u),
				r * sinf (v)
			);
		}

		return vec3 (
			a * cosf (u) * (1.0 + sinf (u)) + r * cosf (u) * cosf (v),
			b * sinf (u) + r * sinf (u) * cosf (v),
			r * sinf (v)
		);
	}

	vec3 norm (float u, float v) {

		// u E [0, pi]
		// dX dU = sin (u) * (a * (-sin (u) - 1)) - c * cos (v) + a * cos^2(u) + c * sin (u) * cos (u) * cos (v)
		// dX dV = 0.5 * c * (cos (u) - 2) * cos (u) * sin (v)
		// dY dU = cos (u) * (b + c * sin^2(u)) - c * sin^2(u) - 0.5 * c * cos^3(u) + c * cos^2(u)
		// dY dV = 0
		// dZ dU = 0.5 * c * sin (u) * sin (v)
		// dZ dV = c * (1.0 - 0.5 * cos (u)) * cos (v)

		// u E [pi, 2pi]
		// dX dU = a * cos^2(u) - sin(u) * (a * (sin(u) + 1) - 0.5 * c * cos (v))
		//      derivative of a * cos (u) * (1.0 + sin (u)) - (c * (1.0 - 0.5 * cos(u))) * cos (v + pi) with respect to u
		// dX dV = 0.5 * c * ( cos (u) - 2 ) * sin (v)
		// dY dU = b * cosf (u)
		// dY dV = 0
		// dZ dU = 0.5 * c * sin (u) * sin (v)
		// dZ dV = c * (1 - 0.5 * cos (u)) * cos (v)

		vec3 du, dv;

		if (u >= M_PI) {
			du = vec3 (
				a * cosf (u) * cosf (u) - sinf (u) * (a * (sinf (u) + 1.0) - 0.5 * c * cosf (v)),
				b * cosf (u),
				0.5 * c * sinf (u) * sinf (v)
			);
			dv = vec3 (
				0.5 * c * ( cosf (u) - 2.0 ) * sinf (v),
				0.0,
				c * (1.0 - 0.5 * cosf (u)) * cosf (v)
			);
		} else {
			du = vec3 (
				sinf (u) * (a * (-sinf (u) - 1.0)) - c * cosf (v) + a * cosf (u) * cosf (u) + c * sinf (u) * cosf (u) * cosf (v),
				cosf (u) * (b + c * sinf (u) * sinf (u)) - c * sinf (u) * sinf (u) - 0.5 * c * cosf (u) * cosf (u) * cosf (u) + c * cosf (u) * cosf (u),
				0.5 * c * sinf (u) * sinf (v)
			);
			dv = vec3 (
				0.5 * c * (cosf (u) - 0.2) * cosf (u) * sinf (v),
				0.0,
				c * (1.0 - 0.5 * cosf (u)) * cosf (v)
			);
		}

		return normalize( cross( du, dv ) ) ;
	}

	VertexData GenVertexData (float u, float v) {

		float _u = u * 2.0 * M_PI;
		float _v = v * 2.0 * M_PI;

		VertexData vd;
		float r = c * ( 1.0 - 0.5 * cosf (_u) );

		if (_u >= M_PI) {

			vd.position = vec3 (
				a * cosf (_u) * (1.0 + sinf (_u)) + r * cosf (_v + M_PI),
				b * sinf (_u),
				r * sinf (_v)
			);

		} else {

			vd.position = vec3 (
				a * cosf (_u) * (1.0 + sinf (_u)) + r * cosf (_u) * cosf (_v),
				b * sinf (_u) + r * sinf (_u) * cosf (_v),
				r * sinf (_v)
			);

		}

		vd.normal = norm (_u, _v);
		vd.texcoord = vec2(u, v);

		return vd;

	}

};