#ifndef __UASTAR_UTIL
#define __UASTAR_UTIL

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
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
using boost::lexical_cast;

#ifndef NO_CPP11
extern std::mt19937 random_engine;
#endif
extern boost::program_options::variables_map vm_options;
extern void help();
extern bool debug;

#define dprintf(fmt, ...) \
    do { if (DEBUG && debug) fprintf(stderr, fmt, __VA_ARGS__); } while (0)
#define dout if (!(DEBUG && debug)) {} else std::cerr

// Suppose we only use x dimension
#define TID (threadIdx.x)
#define GID (THREAD_ID + NT * blockIdx.x)

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
    float maxDiff = 1e-3,
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

#endif
