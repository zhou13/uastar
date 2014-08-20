#ifndef __INPUT_HPP_Y2EQAPM9
#define __INPUT_HPP_Y2EQAPM9

#include "utils.hpp"

class PathwayInput {
public:
    virtual void getStartPoint(int *x, int *y) = 0;
    virtual void getEndPoint(int *x, int *y) = 0;
    virtual void generate(uint8_t graph[]) = 0;
};

#endif /* end of include guard: __INPUT_HPP_Y2EQAPM9 */
