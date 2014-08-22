#include "pathway/input/custom.hpp"

CustomPathwayInput::CustomPathwayInput(int height, int width)
    : m_height(height), m_width(width)
{
}

CustomPathwayInput::~CustomPathwayInput()
{
    // pass
}

void CustomPathwayInput::getStartPoint(int *x, int *y)
{
    *x = m_sx;
    *y = m_sy;
}

void CustomPathwayInput::getEndPoint(int *x, int *y)
{
    *x = m_ex;
    *y = m_ey;
}

void CustomPathwayInput::generate(uint8_t graph[])
{
    cin >> m_sx >> m_sy;
    cin >> m_ex >> m_ey;

    uint8_t *buf = graph;
    for (int i = 0; i < m_height; ++i) {
        for (int j = 0; j < m_width; ++j) {
            int t; cin >> t;
            *buf++ = t ? 0xFF : 0;
        }
    }
}
