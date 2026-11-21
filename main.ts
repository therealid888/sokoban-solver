// Sokoban solver: decompose the level into rooms and solve each room push-optimally.
// Node 22, standard library only. Deterministic: fixed expansion budgets, no clock
// reads, no randomness. The score depends on pushes, so each room is searched in
// PUSHES (A* with a minimum-assignment heuristic) and the rooms are combined.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Expansion budgets (deterministic work caps, never the clock).
const ASTAR_CAP = 150000;
const GREEDY_CAP = 150000;

interface Dir { dx: number; dy: number; ch: string }
const DIRS: Dir[] = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goal: Uint8Array;
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

// Neighbours of cell i inside the floor (walls only). blocked is an optional
// set of cells that count as obstacles (boxes) on top of the walls.
function neighbours(i: number, w: number, h: number, wall: Uint8Array): number[] {
  const x = i % w;
  const y = (i / w) | 0;
  const out: number[] = [];
  if (x > 0 && !wall[i - 1]) out.push(i - 1);
  if (x < w - 1 && !wall[i + 1]) out.push(i + 1);
  if (y > 0 && !wall[i - w]) out.push(i - w);
  if (y < h - 1 && !wall[i + w]) out.push(i + w);
  return out;
}

// BFS from src over floor cells (walls only), avoiding the cells in blocked.
// Returns distances (-1 = unreachable) and parent pointers for walk recovery.
function bfs(src: number, w: number, h: number, wall: Uint8Array, blocked: Set<number>): { dist: Int32Array; parent: Int32Array } {
  const dist = new Int32Array(w * h).fill(-1);
  const parent = new Int32Array(w * h).fill(-1);
  const q: number[] = [src];
  dist[src] = 0;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi];
    for (const n of neighbours(c, w, h, wall)) {
      if (dist[n] !== -1 || blocked.has(n)) continue;
      dist[n] = dist[c] + 1;
      parent[n] = c;
      q.push(n);
    }
  }
  return { dist, parent };
}

// Reconstruct the walk (LURD letters) from src to dst using a BFS parent array.
// The caller guarantees dst is reachable from src.
function walkMoves(src: number, dst: number, parent: Int32Array, w: number): string {
  const steps: string[] = [];
  let c = dst;
  while (c !== src) {
    const p = parent[c];
    const d = c - p;
    steps.push(d === 1 ? 'r' : d === -1 ? 'l' : d === w ? 'd' : 'u');
    c = p;
  }
  steps.reverse();
  return steps.join('');
}

// Connected components of the floor graph with the cells in removed taken out.
function components(w: number, h: number, wall: Uint8Array, removed: Set<number>): number[][] {
  const seen = new Uint8Array(w * h);
  const out: number[][] = [];
  for (let i = 0; i < w * h; i++) {
    if (wall[i] || removed.has(i) || seen[i]) continue;
    const comp: number[] = [];
    const q: number[] = [i];
    seen[i] = 1;
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi];
      comp.push(c);
      for (const n of neighbours(c, w, h, wall)) {
        if (removed.has(n) || seen[n]) continue;
        seen[n] = 1;
        q.push(n);
      }
    }
    out.push(comp);
  }
  return out;
}

interface Room {
  door: number;
  approach: number;
  cells: number[]; // room component cells (excluding the door)
  boxes: number[]; // room boxes (component boxes + the door box, if any)
  goals: number[]; // room goals
}

// Decompose the level into rooms. The hallway is the row holding the player
// start; a door is a floor cell in the row just above or below the hallway,
// adjacent to a hallway floor cell, whose removal separates a component that
// holds boxes/goals and no hallway-row cell. That component plus the door is
// the room; the hallway cell next to the door is the approach.
function findRooms(level: Level): Room[] {
  const { w, h, wall, goal, boxes, player } = level;
  const py = (player / w) | 0;
  const boxSet = new Set(boxes);
  const rooms: Room[] = [];
  const seenRoomKeys = new Set<string>();
  // Candidate door rows: just above and just below the hallway row.
  for (const dy of [-1, 1]) {
    const y = py + dy;
    if (y < 0 || y >= h) continue;
    for (let x = 0; x < w; x++) {
      const d = y * w + x;
      if (wall[d]) continue;
      // The door must open onto a hallway floor cell.
      const approach = py * w + x;
      if (wall[approach]) continue;
      const comps = components(w, h, wall, new Set([d]));
      for (const comp of comps) {
        if (comp.includes(player)) continue;
        // The room component must hold boxes/goals and no hallway-row cell.
        let hasBoxOrGoal = false;
        let hasHallwayCell = false;
        for (const c of comp) {
          if (boxSet.has(c) || goal[c]) hasBoxOrGoal = true;
          if (((c / w) | 0) === py) hasHallwayCell = true;
        }
        if (!hasBoxOrGoal || hasHallwayCell) continue;
        const cells = [...comp].sort((a, b) => a - b);
        const key = cells.join(',');
        if (seenRoomKeys.has(key)) continue;
        seenRoomKeys.add(key);
        const rboxes: number[] = [];
        const rgoals: number[] = [];
        for (const c of comp) {
          if (boxSet.has(c)) rboxes.push(c);
          if (goal[c]) rgoals.push(c);
        }
        if (boxSet.has(d)) rboxes.push(d);
        if (goal[d]) rgoals.push(d);
        rboxes.sort((a, b) => a - b);
        rgoals.sort((a, b) => a - b);
        rooms.push({ door: d, approach, cells, boxes: rboxes, goals: rgoals });
      }
    }
  }
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

// Validate a decomposition: every box and every goal belongs to exactly one
// room, and each room's box count equals its goal count (3 or 4).
function roomsValid(level: Level, rooms: Room[]): boolean {
  if (rooms.length === 0) return false;
  const seenBox = new Set<number>();
  const seenGoal = new Set<number>();
  for (const r of rooms) {
    if (r.boxes.length === 0 || r.boxes.length !== r.goals.length) return false;
    if (r.boxes.length < 3 || r.boxes.length > 4) return false;
    for (const b of r.boxes) {
      if (seenBox.has(b)) return false;
      seenBox.add(b);
    }
    for (const g of r.goals) {
      if (seenGoal.has(g)) return false;
      seenGoal.add(g);
    }
  }
  if (seenBox.size !== level.boxes.length) return false;
  let goalCount = 0;
  for (let i = 0; i < level.goal.length; i++) if (level.goal[i]) goalCount++;
  if (seenGoal.size !== goalCount) return false;
  return true;
}

// ----- room solver -----

interface RoomCtx {
  w: number;
  h: number;
  wall: Uint8Array;
  floor: Set<number>; // room cells + door + approach
  goals: number[];
  goalSet: Set<number>;
  dead: Set<number>; // non-goal cells a box can never leave towards a goal
  distToGoal: Int32Array[]; // distToGoal[gi][cell] = BFS pushes from cell to goal gi
  blocks2x2: number[][]; // each a list of 4 cells (top-left anchored)
}

function buildRoomCtx(level: Level, room: Room): RoomCtx {
  const { w, h, wall } = level;
  const floor = new Set<number>(room.cells);
  floor.add(room.door);
  floor.add(room.approach);
  const goals = room.goals;
  const goalSet = new Set(goals);
  // BFS distances from each goal over the room floor (walls only).
  const distToGoal: Int32Array[] = goals.map((g) => {
    const { dist } = bfs(g, w, h, wall, new Set<number>());
    // Restrict to the room floor: cells outside it are unreachable here.
    for (let i = 0; i < dist.length; i++) if (!floor.has(i)) dist[i] = -1;
    return dist;
  });
  // Dead cells: reverse-reachable from the goals. A box at C could have come
  // from P = C - d when the player could stand at Q = C - 2d (both floor).
  const live = new Set<number>(goals);
  const q: number[] = [...goals];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi];
    const cx = c % w;
    const cy = (c / w) | 0;
    for (const dir of DIRS) {
      const px = cx - dir.dx;
      const py = cy - dir.dy;
      const qx = cx - 2 * dir.dx;
      const qy = cy - 2 * dir.dy;
      if (px < 0 || px >= w || py < 0 || py >= h) continue;
      if (qx < 0 || qx >= w || qy < 0 || qy >= h) continue;
      const p = py * w + px;
      const qq = qy * w + qx;
      if (!floor.has(p) || !floor.has(qq) || live.has(p)) continue;
      live.add(p);
      q.push(p);
    }
  }
  const dead = new Set<number>();
  for (const c of floor) {
    if (!goalSet.has(c) && !live.has(c)) dead.add(c);
  }
  // 2x2 blocks fully inside the room floor's bounding box.
  const blocks2x2: number[][] = [];
  let minX = w, maxX = -1, minY = h, maxY = -1;
  for (const c of floor) {
    const x = c % w, y = (c / w) | 0;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  for (let y = minY; y < maxY; y++) {
    for (let x = minX; x < maxX; x++) {
      blocks2x2.push([y * w + x, y * w + x + 1, (y + 1) * w + x, (y + 1) * w + x + 1]);
    }
  }
  return { w, h, wall, floor, goals, goalSet, dead, distToGoal, blocks2x2 };
}

// Minimum-cost assignment of boxes to goals (brute-force permutations; at
// most 4 boxes). Returns Infinity when no box can reach its goal.
function assignmentHeuristic(ctx: RoomCtx, boxes: number[]): number {
  const k = boxes.length;
  const idx = boxes.map((_, i) => i);
  let best = Infinity;
  // Heap's algorithm for permutations of goal indices.
  const perm = goals0(k);
  const c = new Array<number>(k).fill(0);
  const evalPerm = (p: number[]): void => {
    let s = 0;
    for (let i = 0; i < k; i++) {
      const d = ctx.distToGoal[p[i]][boxes[i]];
      if (d < 0) return;
      s += d;
      if (s >= best) return;
    }
    if (s < best) best = s;
  };
  evalPerm(perm);
  let i = 0;
  while (i < k) {
    if (c[i] < i) {
      if (i % 2 === 0) {
        const t = perm[0]; perm[0] = perm[i]; perm[i] = t;
      } else {
        const t = perm[c[i]]; perm[c[i]] = perm[i]; perm[i] = t;
      }
      evalPerm(perm);
      c[i]++;
      i = 0;
    } else {
      c[i] = 0;
      i++;
    }
  }
  return best;
  function goals0(n: number): number[] {
    const a: number[] = [];
    for (let j = 0; j < n; j++) a.push(j);
    return a;
  }
}

function isDeadlock(ctx: RoomCtx, boxes: number[]): boolean {
  const boxSet = new Set(boxes);
  for (const b of boxes) {
    if (ctx.dead.has(b)) return true;
  }
  for (const block of ctx.blocks2x2) {
    let allBlocked = true;
    let hasOffGoalBox = false;
    for (const c of block) {
      if (ctx.wall[c]) continue;
      if (boxSet.has(c)) {
        if (!ctx.goalSet.has(c)) hasOffGoalBox = true;
        continue;
      }
      allBlocked = false;
      break;
    }
    if (allBlocked && hasOffGoalBox) return true;
  }
  return false;
}

interface RState {
  boxes: number[];
  player: number; // actual player cell
  regionRep: number; // canonical representative of the reachable region
  g: number; // pushes so far
  parent: RState | null;
  moves: string; // moves from parent to this state (walk + push)
}

interface HeapEntry { f: number; seq: number; state: RState }

function heapPush(heap: HeapEntry[], e: HeapEntry): void {
  heap.push(e);
  let i = heap.length - 1;
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (cmpEntry(heap[p], heap[i]) <= 0) break;
    const t = heap[p]; heap[p] = heap[i]; heap[i] = t;
    i = p;
  }
}
function cmpEntry(a: HeapEntry, b: HeapEntry): number {
  if (a.f !== b.f) return a.f - b.f;
  return a.seq - b.seq;
}
function heapPop(heap: HeapEntry[]): HeapEntry | undefined {
  if (heap.length === 0) return undefined;
  const top = heap[0];
  const last = heap.pop() as HeapEntry;
  if (heap.length > 0) {
    heap[0] = last;
    let i = 0;
    for (;;) {
      const l = 2 * i + 1;
      const r = 2 * i + 2;
      let m = i;
      if (l < heap.length && cmpEntry(heap[l], heap[m]) < 0) m = l;
      if (r < heap.length && cmpEntry(heap[r], heap[m]) < 0) m = r;
      if (m === i) break;
      const t = heap[m]; heap[m] = heap[i]; heap[i] = t;
      i = m;
    }
  }
  return top;
}

// Solve one room push-optimally (A*), with a greedy best-first fallback.
// The player starts at the approach cell. Returns the move string (from the
// approach cell), the push count and the player's final cell, or null.
function solveRoom(ctx: RoomCtx, room: Room, cap: number): { moves: string; pushes: number; endPlayer: number } | null {
  const { w, h, wall, floor, goals, goalSet } = ctx;
  const initBoxes = [...room.boxes].sort((a, b) => a - b);
  const goalKey = [...goals].sort((a, b) => a - b).join(',');

  const regionOf = (player: number, boxes: number[]): { region: Set<number>; rep: number; parent: Int32Array } => {
    const blocked = new Set(boxes);
    const { dist, parent } = bfs(player, w, h, wall, blocked);
    const region = new Set<number>();
    let rep = Infinity;
    for (let i = 0; i < dist.length; i++) {
      if (dist[i] >= 0 && floor.has(i)) {
        region.add(i);
        if (i < rep) rep = i;
      }
    }
    return { region, rep, parent };
  };

  const solved = (boxes: number[], region: Set<number>): boolean => {
    if (boxes.join(',') !== goalKey) return false;
    return region.has(room.door);
  };

  const search = (greedy: boolean, budget: number): { moves: string; pushes: number; endPlayer: number } | null => {
    const heap: HeapEntry[] = [];
    const closed = new Map<string, number>();
    let seq = 0;
    let expansions = 0;
    const startRegion = regionOf(room.approach, initBoxes);
    if (isDeadlock(ctx, initBoxes)) return null;
    const h0 = assignmentHeuristic(ctx, initBoxes);
    if (!Number.isFinite(h0)) return null;
    const start: RState = { boxes: initBoxes, player: room.approach, regionRep: startRegion.rep, g: 0, parent: null, moves: '' };
    if (solved(initBoxes, startRegion.region)) return { moves: '', pushes: 0, endPlayer: room.approach };
    heapPush(heap, { f: greedy ? h0 : h0, seq: seq++, state: start });
    closed.set(initBoxes.join(',') + '|' + startRegion.rep, 0);
    while (heap.length > 0) {
      const entry = heapPop(heap) as HeapEntry;
      const s = entry.state;
      const key = s.boxes.join(',') + '|' + s.regionRep;
      if ((closed.get(key) as number) < s.g) continue;
      if (expansions >= budget) return null;
      expansions++;
      const { region, parent } = regionOf(s.player, s.boxes);
      const boxSet = new Set(s.boxes);
      for (const b of s.boxes) {
        const bx = b % w;
        const by = (b / w) | 0;
        for (const dir of DIRS) {
          const behindX = bx - dir.dx;
          const behindY = by - dir.dy;
          const beyondX = bx + dir.dx;
          const beyondY = by + dir.dy;
          if (behindX < 0 || behindX >= w || behindY < 0 || behindY >= h) continue;
          if (beyondX < 0 || beyondX >= w || beyondY < 0 || beyondY >= h) continue;
          const behind = behindY * w + behindX;
          const beyond = beyondY * w + beyondX;
          if (!region.has(behind)) continue;
          if (wall[beyond] || boxSet.has(beyond) || !floor.has(beyond)) continue;
          const nb = s.boxes.map((x) => (x === b ? beyond : x)).sort((a, b) => a - b);
          if (isDeadlock(ctx, nb)) continue;
          const nRegion = regionOf(b, nb);
          const nKey = nb.join(',') + '|' + nRegion.rep;
          const ng = s.g + 1;
          const prevG = closed.get(nKey);
          if (prevG !== undefined && prevG <= ng) continue;
          const nh = assignmentHeuristic(ctx, nb);
          if (!Number.isFinite(nh)) continue;
          const walk = walkMoves(s.player, behind, parent, w);
          const ns: RState = { boxes: nb, player: b, regionRep: nRegion.rep, g: ng, parent: s, moves: walk + dir.ch };
          if (solved(nb, nRegion.region)) {
            // Reconstruct the full move string.
            const parts: string[] = [];
            let cur: RState | null = ns;
            while (cur !== null && cur.parent !== null) {
              parts.push(cur.moves);
              cur = cur.parent;
            }
            parts.reverse();
            return { moves: parts.join(''), pushes: ng, endPlayer: b };
          }
          closed.set(nKey, ng);
          heapPush(heap, { f: greedy ? nh : ng + nh, seq: seq++, state: ns });
        }
      }
    }
    return null;
  };

  const optimal = search(false, cap);
  if (optimal !== null) return optimal;
  return search(true, GREEDY_CAP);
}

// Walk (no pushes) from src to dst on the whole level, avoiding walls and the
// given box cells. Returns the LURD string, or null when unreachable.
function walkWhole(level: Level, src: number, dst: number, boxes: Set<number>): string | null {
  const { w, h, wall } = level;
  const { dist, parent } = bfs(src, w, h, wall, boxes);
  if (dist[dst] < 0) return null;
  return walkMoves(src, dst, parent, w);
}

// Replay a move string on the level to count pushes (and confirm legality).
function replayPushes(level: Level, moves: string): { pushes: number; legal: boolean; solved: boolean } {
  const { w, wall, goal } = level;
  const box = new Uint8Array(wall.length);
  for (const b of level.boxes) box[b] = 1;
  let player = level.player;
  let pushes = 0;
  for (const ch of moves) {
    const dir = DIRS.find((d) => d.ch === ch);
    if (!dir) return { pushes, legal: false, solved: false };
    const d = dir.dy * w + dir.dx;
    const next = player + d;
    if (wall[next]) return { pushes, legal: false, solved: false };
    if (box[next]) {
      const beyond = next + d;
      if (wall[beyond] || box[beyond]) return { pushes, legal: false, solved: false };
      box[next] = 0;
      box[beyond] = 1;
      pushes++;
    }
    player = next;
  }
  for (let i = 0; i < box.length; i++) {
    if (box[i] && !goal[i]) return { pushes, legal: true, solved: false };
  }
  return { pushes, legal: true, solved: true };
}

/** The moves for a level in XSB. */
const solve = (levelText: string): string => {
  const level = parseLevel(levelText);
  const rooms = findRooms(level);
  if (!roomsValid(level, rooms)) return '';
  // Solve rooms in door order; each room is independent.
  const curBoxes = new Set<number>(level.boxes);
  let player = level.player;
  let moves = '';
  for (const room of rooms) {
    const walk = walkWhole(level, player, room.approach, curBoxes);
    if (walk === null) return moves;
    moves += walk;
    player = room.approach;
    const ctx = buildRoomCtx(level, room);
    const sol = solveRoom(ctx, room, ASTAR_CAP);
    if (sol === null) return moves;
    moves += sol.moves;
    // The room's boxes are now on its goals; the player ends at sol.endPlayer.
    for (const b of room.boxes) curBoxes.delete(b);
    for (const g of room.goals) curBoxes.add(g);
    player = sol.endPlayer;
  }
  return moves;
};

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
