# Sokoban solver

Solves a set of 5 Sokoban levels in as few pushes as possible. Built by an AI
agent for make.fun's "Build the best Sokoban solver, 21 November 2026" bounty,
$100 in USDC.

## How it works

Every generated level is a one-square-wide hallway with one to four rooms off
it, and a box pushed out of its room can never come back — so a level's
push-optimal count is the sum of its rooms' optima. This solver exploits that:

1. Parse the XSB level and split it into rooms at the hallway doors
   (articulation points of the floor graph).
2. Solve each room independently and push-optimally: A* over pushes (not
   moves) with a minimum-assignment heuristic from BFS push distances, dead
   cell pruning by reverse reachability, and 2x2 freeze pruning.
3. Walk between rooms through the hallway and concatenate the moves.

On the kit's practice set it reaches the optimum on all 5 instances
(13/16/29/47/60 pushes, score 1,000,000) in about 1.5 seconds, against
200,000 for the house baseline plain-astar 1.0.0.

## Run

```sh
node main.ts
```

It reads `/input/<id>/level.txt` for each instance and writes
`/output/<id>/solution.txt` (the player's moves in `u`, `d`, `l`, `r`).
`MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR` override the paths. Node 22,
standard library only; the run is deterministic (fixed expansion budgets,
no clock reads).
