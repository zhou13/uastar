#define NO_CPP11

#include "puzzle/puzzle.cuh"

#include "puzzle/CPU-solver.hpp"
#include "puzzle/GPU-solver.cuh"

class PuzzlePrivate {
public:
    bool cpuSolved;
    bool cpuSuccessful;
    int cpuOptimal;
    vector<int> cpuSolution;

    bool gpuSolved;
    bool gpuSuccessful;
    int gpuOptimal;
    vector<int> gpuSolution;

    cpusolver::CPUPuzzleSolver<3> *c3;
    cpusolver::CPUPuzzleSolver<4> *c4;
    cpusolver::CPUPuzzleSolver<5> *c5;

    gpusolver::GPUPuzzleSolver<3> *g3;
    gpusolver::GPUPuzzleSolver<4> *g4;
    gpusolver::GPUPuzzleSolver<5> *g5;

    PuzzlePrivate() : c3(0), c4(0), c5(0), g3(0), g4(0), g5(0) {}
    ~PuzzlePrivate() {
        if (c3) delete c3;
        if (c4) delete c4;
        if (c5) delete c5;
        if (g3) delete g3;
        if (g4) delete g4;
        if (g5) delete g5;
    }
};

Puzzle::Puzzle()
{
    if (!vm_options.count("width") && !vm_options.count("height")) {
        cout << "Please set the width or height for your graph." << endl
            << "===============================================" << endl
            << endl;
        help();
    } else if (vm_options.count("width") && vm_options.count("height") &&
               vm_options["width"].as<int>() != vm_options["height"].as<int>()) {
        cout << "Currently we force width must equeal to height." << endl
            << "================================================" << endl
            << endl;
        help();
    } else if (vm_options.count("width")) {
        n = vm_options["width"].as<int>();
    } else if (vm_options.count("height")) {
        n = vm_options["height"].as<int>();
    }

    d = new PuzzlePrivate();
    switch (n) {
    case 3:
        d->c3 = new cpusolver::CPUPuzzleSolver<3>(this);
        d->g3 = new gpusolver::GPUPuzzleSolver<3>(this);
        break;
    case 4:
        d->c4 = new cpusolver::CPUPuzzleSolver<4>(this);
        d->g4 = new gpusolver::GPUPuzzleSolver<4>(this);
        break;
    case 5:
        d->c5 = new cpusolver::CPUPuzzleSolver<5>(this);
        d->g5 = new gpusolver::GPUPuzzleSolver<5>(this);
        break;
    default:
        cout << "Currently we can only solve N=3~5 puzzle problem" << endl
             << "================================================" << endl
             << endl;
        help();
    }
}

Puzzle::~Puzzle()
{
    if (d)
        delete d;
}

string Puzzle::problemName() const
{
    return "Tile Puzzle";
}

void Puzzle::prepare()
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            int t; cin >> t;
            m_initialState.push_back(t);
        }
}

void Puzzle::cpuInitialize()
{
    switch (n) {
    case 3:
        d->c3->initialize();
        break;
    case 4:
        d->c4->initialize();
        break;
    case 5:
        d->c5->initialize();
        break;
    };
}

void Puzzle::gpuInitialize()
{
    switch (n) {
    case 3:
        d->g3->initialize();
        break;
    case 4:
        d->g4->initialize();
        break;
    case 5:
        d->g5->initialize();
        break;
    };
}

void Puzzle::cpuSolve()
{
    switch (n) {
    case 3:
        d->cpuSuccessful = d->c3->solve();
        break;
    case 4:
        d->cpuSuccessful = d->c4->solve();
        break;
    case 5:
        d->cpuSuccessful = d->c5->solve();
        break;
    };
    d->cpuSolved = true;
}

void Puzzle::gpuSolve()
{
    switch (n) {
    case 3:
        d->gpuSuccessful = d->g3->solve();
        break;
    case 4:
        d->gpuSuccessful = d->g4->solve();
        break;
    case 5:
        d->gpuSuccessful = d->g5->solve();
        break;
    };
    d->cpuSolved = true;
}

bool Puzzle::output()
{
    if (d->cpuSolved && d->gpuSolved) {
        if (d->cpuSuccessful != d->gpuSuccessful)
            return false;
    }

    if (d->cpuSolved) {
        if (d->cpuSuccessful) {
            switch (n) {
            case 3:
                d->c3->getSolution(&d->cpuOptimal, &d->cpuSolution);
                break;
            case 4:
                d->c4->getSolution(&d->cpuOptimal, &d->cpuSolution);
                break;
            case 5:
                d->c5->getSolution(&d->cpuOptimal, &d->cpuSolution);
                break;
            }
            printSolution(d->cpuSolution, "solutionCPU.txt");
        } else {
            cout << "No solution from CPU." << endl;
        }
    }

    if (d->gpuSolved) {
        if (d->gpuSuccessful) {
            switch (n) {
            case 3:
                d->g3->getSolution(&d->gpuOptimal, &d->gpuSolution);
                break;
            case 4:
                d->g4->getSolution(&d->gpuOptimal, &d->gpuSolution);
                break;
            case 5:
                d->g5->getSolution(&d->gpuOptimal, &d->gpuSolution);
                break;
            }
            printSolution(d->gpuSolution, "solutionGPU.txt");
        } else {
            cout << "No solution from GPU." << endl;
        }
    }

    if (d->cpuSuccessful) {
        printf(" > Optimal steps from CPU: %d\n", d->cpuOptimal);
    }
    if (d->gpuSuccessful) {
        printf(" > Optimal steps from GPU: %d\n", d->gpuOptimal);
    }

    if (d->cpuSolved && d->gpuSolved) {
        if (d->cpuOptimal != d->gpuOptimal)
            return false;
    }

    return true;
}

void Puzzle::printSolution(const vector<int> &solution, const string &filename) const
{
    FILE *fout = fopen(filename.c_str(), "w");

    if (!fout) {
        printf("ERROR: %s cannot be open for writting.", filename.c_str());
        return;
    }

    vector<uint8_t> state = m_initialState;
    printState(state, fout);

    for (int k = 0; k < (int)solution.size(); ++k) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (state[tileID(i, j)] == 0) {
                    int ni = i + DX[solution[k]];
                    int nj = j + DY[solution[k]];
                    std::swap(state[tileID(i, j)], state[tileID(ni, nj)]);
                    goto finished;
                }
finished:
        printState(state, fout);
    }

    fclose(fout);
}

void Puzzle::printState(const vector<uint8_t> &state, FILE *f) const
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fprintf(f, "%4d", state[i*n+j]);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "\n");
}

