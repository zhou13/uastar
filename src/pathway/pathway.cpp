#include <bitmap_image.hpp>

#include "pathway/pathway.hpp"
#include "pathway/input/custom.hpp"
#include "pathway/input/zigzag.hpp"
#include "pathway/CPU-solver.hpp"
#include "pathway/GPU-solver.hpp"

static void drawPixel(
    bitmap_image &image, int pixel_size,
    int y, int x, uint8_t r, uint8_t g, uint8_t b)
{
    for (int p = 0; p < pixel_size; ++p)
        for (int q = 0; q < pixel_size; ++q)
            image.set_pixel(x * pixel_size + p, y * pixel_size + q, r, g, b);
}

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
    cpuSuccessful = false;
    gpuSuccessful = false;
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
    } else if (m_inputModule == "zigzag") {
        ZigzagPathwayInput input(m_height, m_width);
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
    gpuSolver->initialize();
}

void Pathway::cpuSolve()
{
    cpuSuccessful = cpuSolver->solve();
    cpuSolved = true;
}

void Pathway::gpuSolve()
{
    gpuSuccessful = gpuSolver->solve();
    gpuSolved = true;
}

bool Pathway::output()
{
    if (cpuSolved && gpuSolved) {
        if (cpuSuccessful != gpuSuccessful)
            return false;
    }

    if (cpuSolved) {
        if (cpuSuccessful) {
            cpuSolver->getSolution(&cpuOptimal, &cpuSolution);
            printSolution(cpuSolution, "solutionCPU.txt");
        } else {
            cout << "No solution from CPU." << endl;
        }
    }

    if (gpuSolved) {
        if (gpuSuccessful) {
            gpuSolver->getSolution(&gpuOptimal, &gpuSolution);
            printSolution(gpuSolution, "solutionGPU.txt");
        } else {
            cout << "No solution from GPU." << endl;
        }
    }

    if (cpuSuccessful) {
        printf(" > Optimal distance from CPU: %.3f\n", cpuOptimal);
        plotSolution(cpuSolution, "pathwayCPU.bmp");
    }
    if (gpuSuccessful) {
        printf(" > Optimal distance from GPU: %.3f\n", gpuOptimal);
        plotSolution(gpuSolution, "pathwayGPU.bmp");
    }
    plotSolution(vector<int>(), "pathway.bmp");

    if (cpuSolved && gpuSolved) {
        if (!float_equal(cpuOptimal, gpuOptimal))
            return false;
    }

    return true;
}

void Pathway::generateGraph(PathwayInput &input)
{
    m_graph.clear();
    m_graph.resize(width() * height());
    input.generate(m_graph.data());
    input.getStartPoint(&m_sx, &m_sy);
    input.getEndPoint(&m_ex, &m_ey);
}

void Pathway::printSolution(const vector<int> &pathList,
                            const string filename) const
{
    FILE *fout = fopen(filename.c_str(), "w");

    if (!fout) {
        printf("ERROR: %s cannot be open for writting.", filename.c_str());
        return;
    }

    int px, py;
    int count = 0;
    int count1 = 0;
    int count2 = 0;
    for (int v : pathList) {
        if (count)
            fprintf(fout, " -> ");
        if (++count % 6 == 0) {
            fprintf(fout, "\n\t");
        }
        int x, y;
        toXY(v, &x, &y);
        if (abs(px-x) + abs(py-y) == 2)
            count2++;
        else
            count1++;
        px = x;
        py = y;
        fprintf(fout, "(%d %d)", x, y);
    }

    printf(" > Number of -: %d, +: %d\n", count1, count2);
    fclose(fout);
}

void Pathway::plotSolution(const vector<int> &pathList,
                           const string filename) const
{
    int pixel_size = min(1024 / 3 / width(), 768 / 3 / height());
    if (pixel_size < 4) {
        puts("Warning:  too small pixel to plot");
        return;
    }

    bitmap_image image(width() * 3 * pixel_size, height() * 3 * pixel_size);

    // draw the background
    const uint8_t *buf = graph();
    for (int i = 0; i < height(); ++i)
        for (int j = 0; j < width(); ++j) {
            int x = 3 * i + 1;
            int y = 3 * j + 1;
            drawPixel(image, pixel_size, x, y, 255, 255, 255);
            for (int k = 0; k < 8; ++k) {
                uint8_t col = (*buf & 1 << k) ? 255 : 128;
                drawPixel(image, pixel_size, x+DX[k], y+DY[k], col, col, col);
            }

            ++buf;
        }

    // draw visited nodes
    for (int v : pathList) {
        int x, y;
        toXY(v, &x, &y);
        drawPixel(image, pixel_size, 3*x + 1, 3*y + 1, 0, 255, 0);
    }
    // draw edge
    for (int i = 0; i < (int)pathList.size() - 1; ++i) {
        int ax, ay, bx, by, dx, dy;
        toXY(pathList[i], &ax, &ay);
        toXY(pathList[i+1], &bx, &by);
        dx = ax - bx;
        dy = ay - by;
        drawPixel(image, pixel_size, 3*ax+1-dx, 3*ay+1-dy, 0, 255, 0);
        drawPixel(image, pixel_size, 3*bx+1+dx, 3*by+1+dy, 0, 255, 0);
    }

    image_drawer drawer(image);
    for (int i = 1; i < height() * 3; ++i) {
        if (i % 3 == 0) {
            drawer.pen_color(255, 0, 0);
            drawer.pen_width(3);
        } else {
            drawer.pen_color(0, 0, 255);
            drawer.pen_width(2);
        }
        drawer.horiztonal_line_segment(
            5, width() * 3 * pixel_size - 6, i * pixel_size);
    }
    for (int i = 1; i < width() * 3; ++i) {
        if (i % 3 == 0) {
            drawer.pen_color(255, 0, 0);
            drawer.pen_width(3);
        } else {
            drawer.pen_color(0, 0, 255);
            drawer.pen_width(2);
        }
        drawer.vertical_line_segment(
            5, height() * 3 * pixel_size - 6, i * pixel_size);
    }

    // save image
    image.save_image(filename);
}
