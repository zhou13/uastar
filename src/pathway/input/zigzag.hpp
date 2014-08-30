#ifndef __ZIGZAG_HPP_QPEVYXIS
#define __ZIGZAG_HPP_QPEVYXIS

#include "pathway/input.hpp"

class ZigzagPathwayInput : public PathwayInput {
public:
    ZigzagPathwayInput(int height, int width);
    ~ZigzagPathwayInput();
    void generate(uint8_t graph[]) override;
    void getStartPoint(int *x, int *y);
    void getEndPoint(int *x, int *y);

protected:
    int m_height;
    int m_width;
    int m_sx, m_sy;
    int m_ex, m_ey;
};

#endif /* end of include guard: __ZIGZAG_HPP_QPEVYXIS */
