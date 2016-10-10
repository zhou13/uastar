uAstar
======

Universal General Propose A\* for a GPU Platform

uAstar is a research-proposed program, written in C++, which is able to solve
traditional general-propose A* with GPU acceleration.  Currently, uAstar can
solves two kinds of problems.

1. Find the shortest path on a 8-direction grid network(graph).
   We use a very compact graph representation in a way such that
   we can solve the problem with up to 1x10^9 nodes.
   EXAMPLE (manually input graph through IO):

   ````
   $ ./uastar --pathway -H 5 -W 5 --input-module custom
   > 0 0 4 4     # find path from (0, 0) to (4, 4)
   > 1 0 1 1 1   # 0 represents obstacle
   > 1 0 1 1 1
   > 1 0 1 1 1
   > 1 1 1 0 1
   > 1 1 1 0 1
   ````

   EXAMPLE (random generated graph with 50% paths blocked):
   ````
   $ ./uastar --pathway -H 5 -W 5 --input-module random  --block-rate 50
   ````

2.  Solve the tile puzzle (or sliding puzzle) problem.  The
    "Disjoint pattern database" is used to accelerate the solving
    process.  For really large puzzle problem that tradition A* cannot
    solves with reasonable size of memory, the memory bounded scheme
    is used to fetch the solution without the guarantee of optimality.
