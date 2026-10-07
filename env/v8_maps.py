"""Deterministic object challenges with paired spawns and disjoint map splits.

The layouts are structural fixtures. Importing this module never plays an
episode on a validation or final map.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass

from env.hide_seek_v3_env import DEFAULT_MAP
from env.hide_seek_v7_env import Cell, ObjectLayout
from env.v4_maps import TEST_MAPS as V4_TEST_MAPS
from env.v4_maps import TRAIN_MAPS as V4_TRAIN_MAPS
from env.v4_maps import VALIDATION_MAPS as V4_VALIDATION_MAPS
from env.v5_maps import FINAL_TEST_MAPS as V5_FINAL_TEST_MAPS
from env.v5_maps import TRAIN_MAPS as V5_TRAIN_MAPS
from env.v5_maps import VALIDATION_MAPS as V5_VALIDATION_MAPS
from env.v7_maps import FINAL_LAYOUTS as V7_FINAL_LAYOUTS
from env.v7_maps import TRAIN_LAYOUTS as V7_TRAIN_LAYOUTS
from env.v7_maps import VALIDATION_LAYOUTS as V7_VALIDATION_LAYOUTS


CANONICAL_ROWS: tuple[str, ...] = (
    "##########",
    "#....#...#",
    "#.....#..#",
    "#.....#..#",
    "#...#....#",
    "#....#...#",
    "#....#...#",
    "#....#...#",
    "#....#...#",
    "##########",
)
CANONICAL_ARENA = ObjectLayout(
    CANONICAL_ROWS, block=(5, 3), ramp=(6, 7), low_wall=(5, 7)
)
CANONICAL_SEEKER_SPAWN: Cell = (7, 4)
CANONICAL_HIDER_SPAWN: Cell = (5, 2)

TRAIN_SEED_START = 2_000_000
VALIDATION_SEED_START = 2_300_000
FINAL_SEED_START = 2_200_000

# These optional wall cells sit on room edges away from the block, doorway,
# ramp, landing, and spawn cells. Every candidate is checked after decoration.
_DECORATION_CELLS: tuple[Cell, ...] = tuple(
    (x, y) for x in (1, 8) for y in range(1, 9)
)
_DIRECTIONS: tuple[Cell, ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _component(floors: set[Cell], start: Cell) -> set[Cell]:
    reached = {start}
    pending = deque([start])
    while pending:
        x, y = pending.popleft()
        for dx, dy in _DIRECTIONS:
            neighbor = (x + dx, y + dy)
            if neighbor in floors and neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return reached


@dataclass(frozen=True)
class ChallengeLayout:
    arena: ObjectLayout
    seeker_spawn: Cell
    hider_spawn: Cell

    def validate(self) -> int:
        """Prove the one-push doorway seal and one-ramp entry structurally."""
        size = self.arena.validate()
        floors = {
            (x, y)
            for y, row in enumerate(self.arena.map_rows)
            for x, cell in enumerate(row)
            if cell == "."
        }
        if (
            self.seeker_spawn not in floors
            or self.hider_spawn not in floors
            or self.seeker_spawn == self.hider_spawn
            or self.arena.block in (self.seeker_spawn, self.hider_spawn)
        ):
            raise ValueError("paired spawns must be distinct unoccupied floor cells")
        dx = self.arena.block[0] - self.hider_spawn[0]
        dy = self.arena.block[1] - self.hider_spawn[1]
        doorway = (self.arena.block[0] + dx, self.arena.block[1] + dy)
        if abs(dx) + abs(dy) != 1 or doorway not in floors:
            raise ValueError("Hider must be able to push the block into the doorway")

        # The block at the doorway splits ordinary floor movement. Its only
        # permitted crossing is the marked low wall via the Seeker-side ramp.
        open_floor = floors - {doorway}
        seeker_side = _component(open_floor, self.seeker_spawn)
        hider_side = _component(open_floor, self.arena.block)
        wall = self.arena.low_wall
        ramp = self.arena.ramp
        landing = (2 * wall[0] - ramp[0], 2 * wall[1] - ramp[1])
        if (
            self.hider_spawn not in hider_side
            or hider_side & seeker_side
            or (hider_side | seeker_side) != open_floor
            or ramp not in seeker_side
            or landing not in hider_side
            or len(seeker_side) < 15
            or len(hider_side) < 20
        ):
            raise ValueError("sealed doorway must leave two substantial sides joined by the ramp")
        for step_x, step_y in _DIRECTIONS:
            approach = (doorway[0] - step_x, doorway[1] - step_y)
            beyond = (doorway[0] + step_x, doorway[1] + step_y)
            if approach in seeker_side and beyond in floors:
                raise ValueError("Seeker could reopen the doorway by pushing the block")
        return size


def _transform(cell: Cell, turns: int, reflect: bool, size: int) -> Cell:
    x, y = cell
    if reflect:
        x = size - 1 - x
    for _ in range(turns):
        x, y = size - 1 - y, x
    return x, y


def make_challenge(seed: int) -> ChallengeLayout:
    """Decorate room edges and apply one of the eight square symmetries."""
    rng = random.Random(seed)
    grid = [list(row) for row in CANONICAL_ROWS]
    if seed == TRAIN_SEED_START:
        turns = 0
        reflect = False
    else:
        for x, y in rng.sample(_DECORATION_CELLS, rng.randint(1, 4)):
            grid[y][x] = "#"
        turns = rng.randrange(4)
        reflect = bool(rng.randrange(2))
    size = len(grid)
    transformed = [["#"] * size for _ in range(size)]
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            target_x, target_y = _transform((x, y), turns, reflect, size)
            transformed[target_y][target_x] = value
    rows = tuple("".join(row) for row in transformed)
    place = lambda cell: _transform(cell, turns, reflect, size)
    challenge = ChallengeLayout(
        arena=ObjectLayout(
            rows,
            block=place(CANONICAL_ARENA.block),
            ramp=place(CANONICAL_ARENA.ramp),
            low_wall=place(CANONICAL_ARENA.low_wall),
        ),
        seeker_spawn=place(CANONICAL_SEEKER_SPAWN),
        hider_spawn=place(CANONICAL_HIDER_SPAWN),
    )
    challenge.validate()
    return challenge


_PREVIOUS_ROWS: set[tuple[str, ...]] = {
    DEFAULT_MAP,
    *V4_TRAIN_MAPS.values(), *V4_VALIDATION_MAPS.values(), *V4_TEST_MAPS.values(),
    *V5_TRAIN_MAPS.values(), *V5_VALIDATION_MAPS.values(), *V5_FINAL_TEST_MAPS.values(),
    *(challenge.map_rows for challenge in V7_TRAIN_LAYOUTS.values()),
    *(challenge.map_rows for challenge in V7_VALIDATION_LAYOUTS.values()),
    *(challenge.map_rows for challenge in V7_FINAL_LAYOUTS.values()),
}


def _partition(
    prefix: str, start: int, count: int, used_rows: set[tuple[str, ...]],
) -> dict[str, ChallengeLayout]:
    layouts: dict[str, ChallengeLayout] = {}
    for seed in range(start, start + 20_000):
        try:
            challenge = make_challenge(seed)
        except ValueError:
            continue
        rows = challenge.arena.map_rows
        if rows in used_rows:
            continue
        used_rows.add(rows)
        layouts[f"{prefix}{len(layouts) + 1:03d}"] = challenge
        if len(layouts) == count:
            return layouts
    raise RuntimeError(f"Could not build {count} distinct {prefix} challenge layouts")


_used_rows = set(_PREVIOUS_ROWS)
TRAIN_LAYOUTS = _partition("V8T", TRAIN_SEED_START, 100, _used_rows)
VALIDATION_LAYOUTS = _partition("V8E", VALIDATION_SEED_START, 10, _used_rows)
FINAL_LAYOUTS = _partition("V8F", FINAL_SEED_START, 20, _used_rows)


def layout_hash(layouts: dict[str, ChallengeLayout]) -> str:
    """Hash each named layout, objects, and paired spawn cells in name order."""
    payload = [
        (
            name, layout.arena.map_rows, layout.arena.block,
            layout.arena.ramp, layout.arena.low_wall,
            layout.seeker_spawn, layout.hider_spawn,
        )
        for name, layout in sorted(layouts.items())
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
