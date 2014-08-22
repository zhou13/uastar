#ifndef __CUSTOM_HPP_JWMXQYT3
#define __CUSTOM_HPP_JWMXQYT3

#include "pathway/input.hpp"

class CustomPathwayInput : public PathwayInput {
public:
    CustomPathwayInput(int height, int width);
    ~CustomPathwayInput();
    void getStartPoint(int *x, int *y);
    void getEndPoint(int *x, int *y);
    void generate(uint8_t graph[]) override;

protected:
    int m_height;
    int m_width;
    int m_sx, m_sy;
    int m_ex, m_ey;
};

#endif /* end of include guard: __CUSTOM_HPP_JWMXQYT3 */
