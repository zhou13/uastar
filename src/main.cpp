#include <chrono>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include "utils.hpp"
#include "pathway/pathway.hpp"
#include "puzzle/puzzle.hpp"
#include "problem.hpp"

const char *program_description =
R"(uAstar is a research-proposed program, written in C++, that is able to
solve traditional general-propose A* with GPU acceleration.  Currently,
uAstar can solves two kinds of problems.

(1)  Find the shortest path on a 8-direction grid network(graph).
     We use a very compact graph representation in a way such that
     we can solve the problem with up to 1x10^9 nodes.
     
     EXAMPLE (manually input graph through IO):
         ./uastar --pathway -H 5 -W 5 --input-module custom
         > 0 0 4 4     # find path from (0, 0) to (4, 4)
         > 1 0 1 1 1   # 0 represents obstacle
         > 1 0 1 1 1
         > 1 0 1 1 1
         > 1 1 1 0 1
         > 1 1 1 0 1

     EXAMPLE (random generated graph with 50% paths blocked):
         ./uastar --pathway -H 5 -W 5 --input-module random  --block-rate 50

(2)  Solve the tile puzzle (or sliding puzzle) problem.  The
     ``Disjoint pattern database'' is used to accelerate the solving
     process.  For really large puzzle problem that tradition A* cannot
     solves with reasonable size of memory, the memory bounded scheme
     is used to fetch the solution without the guarantee of optimality.

)";

namespace po = boost::program_options;
static po::options_description desc("uAstar Options");

po::variables_map vm_options;
std::mt19937 random_engine;
bool debug;

void help()
{
    cout << program_description << desc << endl;
    exit(1);
}

static string time_pass(std::chrono::steady_clock::time_point start_time)
{
    using namespace std::chrono;
    auto elapse = duration_cast<milliseconds>(steady_clock::now() - start_time);
    string time = (boost::format("%07i") % elapse.count()).str();
    return "[" + time + "]";
}

static void solve_problem(Problem &problem)
{
    auto start_time = std::chrono::steady_clock::now();
    bool use_cpu = !vm_options.count("no-cpu");
    bool use_gpu = !vm_options.count("no-gpu");

    cout << time_pass(start_time)
         << " Generating input data ......"
         << endl;
    problem.prepare();

    if (use_cpu) {
        cout << time_pass(start_time)
            << " Initializing CPU data structure ......"
            << endl;
        problem.cpuInitialize();
    }

    if (use_gpu) {
        cout << time_pass(start_time)
             << " Initializing GPU data structure ......"
             << endl;
        problem.gpuInitialize();
    }

    if (use_cpu) {
        cout << time_pass(start_time)
             << " Solving the problem on a pure CPU platform ......"
             << endl;
        problem.cpuSolve();
    }

    if (use_gpu) {
        cout << time_pass(start_time)
             << " Solving the problem with GPU acceleration ......"
             << endl;
        problem.gpuSolve();
    }

    if (use_cpu || use_gpu) {
        cout << time_pass(start_time)
             << " Checking the result ......"
             << endl;
        if (!problem.output()) {
            cout << time_pass(start_time)
                 << "ERROR: Output of the CPU and GPU is not consistent!"
                 << endl;
            exit(1);
        }
    }
}

int main(int argc, char *argv[])
{
    const char *env_debug = getenv("DEBUG");
    debug = !!env_debug;

    desc.add_options()
        ("help,h", "Print usage message")
        ("pathway", "Solve pathway finding problem")
        ("puzzle", "Solve tile puzzle problem")
        ("height,H", po::value<int>(), "Width of the problem")
        ("width,W", po::value<int>(), "Height of the problem")
        ("input-module", po::value<string>()->default_value("custom"),
         "Choose how to generate the input data.\n"
         "For pathway finding:\n"
         "    custom    -- Fetch the graph from system IO\n"
         "    random    -- Random generated graph with 50\%\n"
         "                 paths blocked"
         "\n"
         "For tile puzzle:\n"
         "    custom -- Fetch the problem from system IO\n"
         )
        ("plot,p", po::value<string>(),
         "Plot the path to a BMP image (only for --pathway)")
        ("block-rate,b", po::value<int>(),
         "Set the block rate (1-99) (only for random module)")
        ("no-cpu,G", "Do not run sequential CPU-based A* search")
        ("no-gpu,C", "Do not run GPU-accelerated A* search")
        ("ordinal,o", po::value<int>(0), "Specify CUDA Ordinal")
        ("seed,s", po::value<int>(), "Random seed of this run")
        ;

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm_options);
        po::notify(vm_options);
    } catch (std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Exception of unknown type!\n";
    }

    if (argc == 1 || vm_options.count("help")) {
        help();
    }
    if (vm_options.count("pathway") + vm_options.count("puzzle") != 1) {
        cout << "Please select the type of problem you want to solve!" << endl
             << "====================================================" << endl
             << endl;
        help();
    }
    if (vm_options.count("seed"))
        random_engine.seed(vm_options["seed"].as<int>());

    if (vm_options.count("pathway")) {
        Pathway pathway;
        solve_problem(pathway);
    }
    if (vm_options.count("puzzle")) {
        Puzzle puzzle;
        solve_problem(puzzle);
    }

    return 0;
}
