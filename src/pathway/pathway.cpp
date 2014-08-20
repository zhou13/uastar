#include "pathway/pathway.hpp"
#include "pathway/input/custom.hpp"
#include "pathway/CPU-solver.hpp"
#include "pathway/GPU-solver.hpp"

Pathway::Pathway()
{
    if (!vm_options.count("width") || !vm_options.count("height")) {
        cout << "Please set the width and height for your graph." << endl
             << "===============================================" << endl
             << endl;
        help();
    }

    m_width = vm_options["width"].as<int>();
    m_height = vm_options["height"].as<int>();
    m_inputModule = vm_options["input-module"].as<string>();
    m_size = m_width * m_height;
    cpuSolver = new CPUPathwaySolver(this);
    gpuSolver = new GPUPathwaySolver(this);
    cpuSolved = false;
    gpuSolved = false;
}

Pathway::~Pathway()
{
    delete cpuSolver;
    delete gpuSolver;
}

string Pathway::problemName() const
{
    return "Pathway Finding";
}

void Pathway::prepare()
{
    m_graph.resize(m_size);
    if (m_inputModule == "custom") {
        CustomPathwayInput input(m_height, m_width);
        generateGraph(input);
    } else {
        cout << "Please set your input-module parameter correctly." << endl
             << "=================================================" << endl
             << endl;;
        help();
    }
}

void Pathway::cpuInitialize()
{
    cpuSolver->initialize();
}

void Pathway::gpuInitialize()
{
    throw runtime_error("Not implemented");
}

void Pathway::cpuSolve()
{
    cpuSuccessful = cpuSolver->solve(&cpuOptimal, &cpuSolution);
    cpuSolved = true;
}

void Pathway::gpuSolve()
{
    throw runtime_error("Not implemented");
}

bool Pathway::output() const
{
    if (cpuSolved && gpuSolved) {
    }
    return true;
}

void Pathway::generateGraph(PathwayInput &input)
{
    input.getStartPoint(&m_sx, &m_sy);
    input.getEndPoint(&m_ex, &m_ey);
    input.generate(m_graph.data());
}
