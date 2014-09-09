#ifndef __STORAGE_HPP_QW7SJEYO
#define __STORAGE_HPP_QW7SJEYO

#include <utils.hpp>

template <int N>
class PuzzleStorage {
    __cuda__ PuzzleStorage(uint8_t input[N][N]);
    __cuda__ void decompose(uint8_t output[3][3]);
    __cuda__ int64_t hashValue() const;
    __cuda__ bool operator==(const PuzzleStorage<3> &r) const;
};

template <int N>
void printStorage(const PuzzleStorage<N> &ps, const string &prefix)
{
    if (DEBUG_CONDITION) {
        uint8_t buf[N][N];
        ps.decompose(buf);
        for (int i = 0; i < N; ++i) {
            cout << prefix;
            for (int j = 0; j < N; ++j)
                printf("%5d", (int)buf[i][j]);
            cout << endl;
        }
    }
}

#ifdef __CUDACC__
template <int N>
__device__ void printStorageDevice(const PuzzleStorage<N> &ps, const char *prefix)
{
        uint8_t buf[N][N];
        ps.decompose(buf);
        for (int i = 0; i < N; ++i) {
            printf(prefix);
            for (int j = 0; j < N; ++j)
                printf("%5d", (int)buf[i][j]);
            printf("\n");
        }
}
#endif

template <>
class PuzzleStorage<3> {
public:
    __cuda__ PuzzleStorage() {}
    __cuda__ PuzzleStorage(uint8_t input[3][3]) {
        bit =
            (input[0][0] << 0 ) +
            (input[0][1] << 4 ) +
            (input[0][2] << 8 ) +
            (input[1][0] << 12) +
            (input[1][1] << 16) +
            (input[1][2] << 20) +
            (input[2][0] << 24) +
            (input[2][1] << 28);
    }

    inline __cuda__ void decompose(uint8_t output[3][3]) const {
        output[0][0] = (bit >> 0 ) & 15;
        output[0][1] = (bit >> 4 ) & 15;
        output[0][2] = (bit >> 8 ) & 15;
        output[1][0] = (bit >> 12) & 15;
        output[1][1] = (bit >> 16) & 15;
        output[1][2] = (bit >> 20) & 15;
        output[2][0] = (bit >> 24) & 15;
        output[2][1] = (bit >> 28) & 15;
        output[2][2] = 36
            - output[0][0] - output[0][1] - output[0][2]
            - output[1][0] - output[1][1] - output[1][2]
            - output[2][0] - output[2][1];
    }

    inline __cuda__ uint64_t hashValue() const {
        return bit;
    }
    inline __cuda__ bool operator==(const PuzzleStorage<3> &r) const {
        return bit == r.bit;
    }

private:
    uint32_t bit;
};

template <>
class PuzzleStorage<4> {
public:
    __cuda__ PuzzleStorage() {}
    __cuda__ PuzzleStorage(uint8_t input[4][4]) {
        bit =
            ((uint64_t)input[0][0] << 0 ) +
            ((uint64_t)input[0][1] << 4 ) +
            ((uint64_t)input[0][2] << 8 ) +
            ((uint64_t)input[0][3] << 12) +
            ((uint64_t)input[1][0] << 16) +
            ((uint64_t)input[1][1] << 20) +
            ((uint64_t)input[1][2] << 24) +
            ((uint64_t)input[1][3] << 28) +
            ((uint64_t)input[2][0] << 32) +
            ((uint64_t)input[2][1] << 36) +
            ((uint64_t)input[2][2] << 40) +
            ((uint64_t)input[2][3] << 44) +
            ((uint64_t)input[3][0] << 48) +
            ((uint64_t)input[3][1] << 52) +
            ((uint64_t)input[3][2] << 56) +
            ((uint64_t)input[3][3] << 60);
    }

    inline __cuda__ void decompose(uint8_t output[4][4]) const {
        output[0][0] = (bit >> 0 ) & 15;
        output[0][1] = (bit >> 4 ) & 15;
        output[0][2] = (bit >> 8 ) & 15;
        output[0][3] = (bit >> 12) & 15;
        output[1][0] = (bit >> 16) & 15;
        output[1][1] = (bit >> 20) & 15;
        output[1][2] = (bit >> 24) & 15;
        output[1][3] = (bit >> 28) & 15;
        output[2][0] = (bit >> 32) & 15;
        output[2][1] = (bit >> 36) & 15;
        output[2][2] = (bit >> 40) & 15;
        output[2][3] = (bit >> 44) & 15;
        output[3][0] = (bit >> 48) & 15;
        output[3][1] = (bit >> 52) & 15;
        output[3][2] = (bit >> 56) & 15;
        output[3][3] = (bit >> 60) & 15;
    }

    inline __cuda__ uint64_t hashValue() const {
        return bit;
    }

    inline __cuda__ bool operator==(const PuzzleStorage<4> &r) const {
        return bit == r.bit;
    }

private:
    uint64_t bit;
};

template <>
class PuzzleStorage<5> {
public:
    __cuda__ PuzzleStorage() {}
    __cuda__ PuzzleStorage(uint8_t input[5][5]) {
        bit1 =
            ((uint64_t)input[0][0] << 0 ) +
            ((uint64_t)input[0][1] << 5 ) +
            ((uint64_t)input[0][2] << 10) +
            ((uint64_t)input[0][3] << 15) +
            ((uint64_t)input[0][4] << 20) +
            ((uint64_t)input[1][0] << 25) +
            ((uint64_t)input[1][1] << 30) +
            ((uint64_t)input[1][2] << 35) +
            ((uint64_t)input[1][3] << 40) +
            ((uint64_t)input[1][4] << 45) +
            ((uint64_t)input[2][0] << 50) +
            ((uint64_t)input[2][1] << 55) +
            (((uint64_t)input[4][4] & 15) << 60);

        bit2 =
            ((uint64_t)input[2][2] << 0 ) +
            ((uint64_t)input[2][3] << 5 ) +
            ((uint64_t)input[2][4] << 10) +
            ((uint64_t)input[3][0] << 15) +
            ((uint64_t)input[3][1] << 20) +
            ((uint64_t)input[3][2] << 25) +
            ((uint64_t)input[3][3] << 30) +
            ((uint64_t)input[3][4] << 35) +
            ((uint64_t)input[4][0] << 40) +
            ((uint64_t)input[4][1] << 45) +
            ((uint64_t)input[4][2] << 50) +
            ((uint64_t)input[4][3] << 55) +
            (((uint64_t)input[4][4] & 16) << 56);
    }

    inline __cuda__ void decompose(uint8_t output[5][5]) const {
        output[0][0] = (bit1 >> 0 ) & 31;
        output[0][1] = (bit1 >> 5 ) & 31;
        output[0][2] = (bit1 >> 10) & 31;
        output[0][3] = (bit1 >> 15) & 31;
        output[0][4] = (bit1 >> 20) & 31;
        output[1][0] = (bit1 >> 25) & 31;
        output[1][1] = (bit1 >> 30) & 31;
        output[1][2] = (bit1 >> 35) & 31;
        output[1][3] = (bit1 >> 40) & 31;
        output[1][4] = (bit1 >> 45) & 31;
        output[2][0] = (bit1 >> 50) & 31;
        output[2][1] = (bit1 >> 55) & 31;

        output[2][2] = (bit2 >> 0 ) & 31;
        output[2][3] = (bit2 >> 5 ) & 31;
        output[2][4] = (bit2 >> 10) & 31;
        output[3][0] = (bit2 >> 15) & 31;
        output[3][1] = (bit2 >> 20) & 31;
        output[3][2] = (bit2 >> 25) & 31;
        output[3][3] = (bit2 >> 30) & 31;
        output[3][4] = (bit2 >> 35) & 31;
        output[4][0] = (bit2 >> 40) & 31;
        output[4][1] = (bit2 >> 45) & 31;
        output[4][2] = (bit2 >> 50) & 31;
        output[4][3] = (bit2 >> 55) & 31;

        output[4][4] = (bit1 >> 60 & 15) + (bit2 >> 56 & 16);
    }

    inline __cuda__ uint64_t hashValue() const {
        return (bit1 ^ bit2) + (bit2 << 2) + (bit1 << 1);
    }

    inline __cuda__ bool operator==(const PuzzleStorage<5> &r) const {
        return bit1 == r.bit1 && bit2 == r.bit2;
    }

private:
    uint64_t bit1, bit2;
};


namespace boost {
    template<int N>
    size_t hash_value(const PuzzleStorage<N> &ps) {
        return ps.hashValue();
    }
}

#endif /* end of include guard: __STORAGE_HPP_QW7SJEYO */

