#include "pathway/input/zigzag.hpp"

ZigzagPathwayInput::ZigzagPathwayInput(int height, int width)
    : m_height(height), m_width(width)
{
    // pass
}

ZigzagPathwayInput::~ZigzagPathwayInput()
{
    // pass
}

void ZigzagPathwayInput::generate(uint8_t graph[])
{
    m_sx = 0;
    m_sy = m_width / 2;
    m_ex = m_height - 1;
    m_ey = m_width / 2;

    int heightGap;
    cin >> heightGap;

    bool left = false;
    uint8_t *buf = graph;
    for (int i = 0; i < m_height; ++i) {
        if (i && i % heightGap == 0) {
            if (left)
                *buf++ = 0xFF;
            for (int j = 0; j < m_width-1; ++j)
                *buf++ = 0;
            if (!left)
                *buf++ = 0xFF;
            left = !left;
        } else {
            for (int j = 0; j < m_width; ++j)
                *buf++ = 0xFF;
        }
    }
}

void ZigzagPathwayInput::getStartPoint(int *x, int *y)
{
    *x = m_sx;
    *y = m_sy;
}

void ZigzagPathwayInput::getEndPoint(int *x, int *y)
{
    *x = m_ex;
    *y = m_ey;
}
