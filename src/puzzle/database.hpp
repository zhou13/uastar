#ifndef __DATABASE_HPP_RJVH1OLD
#define __DATABASE_HPP_RJVH1OLD

#include "utils.hpp"

const int MAX_CONF = 5*5+1;

// Disjoint pattern databse generator
class PatternDatabase {
public:
    PatternDatabase(int n, const vector<int> &tracked);
    virtual ~PatternDatabase();

    size_t size();
    void genDatabase(uint8_t out[]);
    void fetchDatabase(uint8_t out[]);

    uint32_t encoding(const uint8_t in[]);
    void decoding(uint32_t code, uint8_t out[]);

private:
    bool inrange(int x, int y);
    int tileID(int x, int y);
    void tileXY(int id, int *x, int *y);

    int n;
    size_t m_size;
    vector<int> m_tracked;
    vector<int> m_map;  // from number to position in m_tracked
    vector<uint32_t> m_multiple;
};

#endif /* end of include guard: __DATABASE_HPP_RJVH1OLD */

