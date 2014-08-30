#ifndef __UASTAR_UTIL
#define __UASTAR_UTIL

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

// workaround to fix bug https://svn.boost.org/trac/boost/ticket/9392
#if defined(__CUDACC__)
#define BOOST_NOINLINE __attribute__ ((noinline))
#endif

#include <boost/program_options/variables_map.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>

// C++11 is not well supported by CUDA 5.5
#ifndef NO_CPP11
#include <random>
#endif

using std::cin;
using std::cout;
using std::flush;
using std::min;
using std::max;
using std::endl;
using std::pair;
using std::runtime_error;
using std::string;
using std::vector;
using std::make_pair;
using std::numeric_limits;
using boost::lexical_cast;

#ifndef NO_CPP11
extern boost::mt19937 random_engine;
#endif
extern boost::program_options::variables_map vm_options;
extern void help();
extern bool debug;

#define DEBUG_CONDITION (DEBUG && debug)
#define dprintf(fmt, ...) \
    do { \
        if (DEBUG_CONDITION) { \
            fprintf(stderr, fmt, __VA_ARGS__); \
            fflush(stderr); \
        } \
    } while (0)

#define dout if (!DEBUG_CONDITION) {} else std::cerr

union Float_t
{
    Float_t(float num = 0.0f) : f(num) {}
    // Portable extraction of components.
    bool Negative() const { return (i >> 31) != 0; }
    int32_t RawMantissa() const { return i & ((1 << 23) - 1); }
    int32_t RawExponent() const { return (i >> 23) & 0xFF; }

    int32_t i;
    float f;
    struct
    {   // Bitfields for exploration. Do not use in production code.
        uint32_t mantissa : 23;
        uint32_t exponent : 8;
        uint32_t sign : 1;
    } parts;
};

inline bool float_equal(
    float A,
    float B,
    float maxDiff = 1e-3f,
    int maxUlpsDiff = 200)
{
    // Check if the numbers are really close -- needed
    // when comparing numbers near zero.
    float absDiff = fabs(A - B);
    if (absDiff <= maxDiff)
        return true;

    Float_t uA(A);
    Float_t uB(B);

    // Different signs means they do not match.
    if (uA.Negative() != uB.Negative())
        return false;

    // Find the difference in ULPs.
    int ulpsDiff = abs(uA.i - uB.i);
    if (ulpsDiff <= maxUlpsDiff)
        return true;

    return false;
}

#ifdef __CUDACC__
#  define CUDA_FUNC __host__ __device__
#  define CUDA_KERNEL __global__
#else
#  define CUDA_FUNC
#  define CUDA_KERNEL
#  define CUDA_SHARED
#endif

const float SQRT2 = 1.4142135623731f;
const int DX[8] = { 1,  1, -1, -1,  1, -1,  0,  0 };
const int DY[8] = { 1, -1,  1, -1,  0,  0,  1, -1 };
const float COST[8] = {SQRT2, SQRT2, SQRT2, SQRT2, 1, 1, 1, 1};

#endif
