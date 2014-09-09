#include "puzzle/database.hpp"
#include "boost/dynamic_bitset.hpp"
#include "boost/filesystem.hpp"

#include <queue>
#include <bitset>

PatternDatabase::PatternDatabase(int n, const vector<int> &tracked)
    : n(n), m_tracked(tracked)
{
    m_size = 1;
    m_map.resize(n*n);
    m_multiple.reserve(tracked.size()+1);
    for (int i = 0; i < (int)tracked.size(); ++i) {
        m_map[tracked[i]] = i;
        m_size *= (n*n - i);
        m_multiple.push_back(n*n - i);
    }
    m_multiple.push_back(1);
    for (int i = m_multiple.size() - 2; i >= 0; --i)
        m_multiple[i] *= m_multiple[i+1];
    for (int v : m_multiple)
        dout << v << " ";
    dout << endl;
}

PatternDatabase::~PatternDatabase()
{
    // pass
}

size_t PatternDatabase::size()
{
    return m_size;
}

void PatternDatabase::genDatabase(uint8_t out[])
{
    uint8_t conf[MAX_CONF];
    std::queue<uint32_t> q;
    boost::dynamic_bitset<> visited(size());

    memset(conf, 0, sizeof(conf));
    memset(out, 0xFF, m_size);

    for (int v : m_tracked)
        conf[v-1] = v;
    uint32_t code = encoding(conf);
    q.push(code);
    out[code] = 0;
    visited[code] = 1;

    int count = 1;
    while (!q.empty()) {
        uint32_t code = q.front(); q.pop();
        decoding(code, conf);
        for (int x = 0; x < n; ++x)
            for (int y = 0; y < n; ++y) {
                int id = tileID(x, y);
                if (conf[id]) {
                    for (int k = 0; k < 4; ++k) {
                        int nx = x + DX[k];
                        int ny = y + DY[k];
                        int nid = tileID(nx, ny);
                        if (inrange(nx, ny) && conf[nid] == 0) {
                            std::swap(conf[id], conf[nid]);
                            uint32_t ncode = encoding(conf);
                            if (!visited[ncode]) {
                                visited[ncode] = 1;
                                q.push(ncode);
                                out[ncode] = out[code] + 1;
                                ++count;
                                if (count % 10000000 == 0)
                                    cout << "\t"
                                         << count/1024/1024 << "M/"
                                         << m_size/1024/1024 << "M" << endl;
                            }
                            std::swap(conf[id], conf[nid]);
                        }
                    }
                }
            }
    }
}

void PatternDatabase::fetchDatabase(uint8_t out[])
{
    string filename = "database_" + std::to_string(n);
    for (int i = 0; i < (int)m_tracked.size(); ++i)
        filename += "_" + std::to_string(m_tracked[i]);
    filename += ".bin";

    if (boost::filesystem::exists(filename)) {
        FILE *fin = fopen(filename.c_str(), "rb");
        if (!fin) {
            cout << "Cannot read file " << filename << endl;
            exit(1);
        }
        size_t read = fread(out, 1, m_size, fin);
        if (read != m_size) {
            boost::filesystem::remove(filename);
            cout << filename << "is not intact" << endl;
            exit(1);
        }
        fclose(fin);
    } else {
        cout << "\tGenerating pattern databse" << endl;
        genDatabase(out);
        FILE *fout = fopen(filename.c_str(), "wb");
        if (!fout) {
            cout << "Cannot write to file " << filename << endl;
            exit(1);
        }
        size_t write = fwrite(out, 1, m_size, fout);
        if (write != m_size) {
            boost::filesystem::remove(filename);
            cout << filename << "cannot be written (disk is full?)" << endl;
            exit(1);
        }
        fclose(fout);
    }
}

bool PatternDatabase::inrange(int x, int y)
{
    return 0 <= x && x < n && 0 <= y && y < n;
}

int PatternDatabase::tileID(int x, int y)
{
    return x * n + y;
}

void PatternDatabase::tileXY(int id, int *x, int *y)
{
    *x = id / n;
    *y = id % n;
}

uint32_t PatternDatabase::encoding(const uint8_t in[])
{
    static vector<uint8_t> index, vrank;

    if (index.size() != m_tracked.size()) {
        index.resize(m_tracked.size());
        vrank.resize(m_tracked.size());
    }

    uint32_t retn = 0;
    int nn = n*n;
    for (int i = 0; i < nn; ++i)
        if (in[i]) {
            index[m_map[in[i]]] = i;
        }
    copy(index.begin(), index.end(), vrank.begin());
    for (int i = 0; i < (int)index.size(); ++i)
        for (int j = i+1; j < (int)index.size(); ++j)
            if (index[i] < index[j])
                --vrank[j];
    for (int i = 0; i < (int)vrank.size(); ++i)
        retn += vrank[i] * m_multiple[i+1];
    return retn;
}

void PatternDatabase::decoding(uint32_t code, uint8_t out[])
{
    static vector<uint8_t> value;
    static vector<bool> used;

    if (value.size() != m_tracked.size())
        value.resize(m_tracked.size());
    if ((int)used.size() != n*n)
        used.resize(n*n);

    used.assign(n*n, false);

    for (int i = 0; i < (int)m_tracked.size(); ++i) {
        value[i] = code / m_multiple[i+1];
        code %= m_multiple[i+1];
    }

    for (int i = 0; i < (int)m_tracked.size(); ++i) {
        int cnt = 0;
        while (used[cnt])
            ++cnt;
        for (int j = 0; j < value[i]; ++j) {
            ++cnt;
            while (used[cnt])
                ++cnt;
        }
        value[i] = cnt;
        used[cnt] = true;
    }

    memset(out, 0, n*n);
    for (int i = 0; i < (int)value.size(); ++i) {
        out[value[i]] = m_tracked[i];
    }
}
