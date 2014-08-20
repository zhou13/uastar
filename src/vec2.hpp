#ifndef __VEC2_HPP_PIDGL0CG
#define __VEC2_HPP_PIDGL0CG

#include "utils.hpp"

class vec2 {
public:
    union {
        struct {
            real x, y;
        };
        real f[2];
    };

    vec2(): x(0), y(0) { }
    vec2(real x, real y): x(x), y(y) { }
    vec2 operator +(const vec2 &r) const {
        return vec2(x + r.x, y + r.y);
    }
    vec2 &operator +=(const vec2 &r) {
        x += r.x;
        y += r.y;
        return *this;
    }
    vec2 operator -(const vec2 &r) const {
        return vec2(x + r.x, y + r.y);
    }
    vec2 &operator -=(const vec2 &r) {
        x -= r.x;
        y -= r.y;
        return *this;
    }
    vec2 operator -() const {
        return vec2(-x, -y);
    }
    vec2 operator *(real r) const {
        return vec2(x * r, y * r);
    }
    vec2 &operator *=(real r) {
        x *= r;
        y *= r;
        return *this;
    }
    vec2 operator /(real r) const {
        return vec2(x / r, y / r);
    }
    vec2 &operator /=(real r) {
        x *= r;
        y *= r;
        return *this;
    }
    real dot(const vec2 &r) const {
        return x*r.x + y*r.y;
    }
    real cross(const vec2 &r) const {
        return x*r.y - y*r.x;
    }
    real length() const {
        return sqrt(x*x + y*y);
    }
    real length2() const {
        return x*x + y*y;
    }
    real distance(const vec2 &r) const {
        return (r-*this).length();
    }
    real distance2(const vec2 &r) const {
        return (r-*this).length();
    }
    vec2 norm() const {
        return *this / length();
    }
};

inline vec2 operator*(float f, const vec2 &v) {
    return v*f;
}

#endif /* end of include guard: __VEC2_HPP_PIDGL0CG */
