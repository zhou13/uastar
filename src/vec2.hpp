#ifndef __VEC2_HPP_PIDGL0CG
#define __VEC2_HPP_PIDGL0CG

#include "utils.hpp"

template <typename T> inline T sqr(T x) { return x*x; }

class vec2 {
public:
    int x, y;

    vec2(): x(0), y(0) { }
    vec2(int x, int y): x(x), y(y) { }
    vec2 operator +(const vec2 &r) const {
        return vec2(x + r.x, y + r.y);
    }
    vec2 &operator +=(const vec2 &r) {
        x += r.x;
        y += r.y;
        return *this;
    }
    vec2 operator -(const vec2 &r) const {
        return vec2(x - r.x, y - r.y);
    }
    vec2 &operator -=(const vec2 &r) {
        x -= r.x;
        y -= r.y;
        return *this;
    }
    vec2 operator -() const {
        return vec2(-x, -y);
    }
    vec2 operator *(int r) const {
        return vec2(x * r, y * r);
    }
    vec2 &operator *=(int r) {
        x *= r;
        y *= r;
        return *this;
    }
    vec2 operator /(int r) const {
        return vec2(x / r, y / r);
    }
    vec2 &operator /=(int r) {
        x *= r;
        y *= r;
        return *this;
    }

    float distance(const vec2 &r) { return (float)sqrt(sqr(r.x - x) + sqr(r.y - y));}

    string str() const {
        return 
            "(" + boost::lexical_cast<string>(x) +
            "," + boost::lexical_cast<string>(y) + ")";
    }

    friend std::ostream &operator<<(std::ostream &o, const vec2 &v) {
        o << v.str();
        return o;
    }
};

#endif /* end of include guard: __VEC2_HPP_PIDGL0CG */
