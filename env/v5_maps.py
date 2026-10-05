"""Deterministic procedural map splits reserved for V5 experiments."""

from __future__ import annotations

import random
from collections.abc import Iterable

from env.hide_seek_v3_env import DEFAULT_MAP, HideSeekV3Env
from env.v4_maps import TEST_MAPS as V4_TEST_MAPS
from env.v4_maps import TRAIN_MAPS as V4_TRAIN_MAPS
from env.v4_maps import VALIDATION_MAPS as V4_VALIDATION_MAPS


MapRows = tuple[str, ...]

TRAIN_SEED_RANGE = range(500_000, 510_000)
VALIDATION_SEED_RANGE = range(600_000, 610_000)
FINAL_TEST_SEED_RANGE = range(700_000, 710_000)

TRAIN_MAP_COUNT = 100
VALIDATION_MAP_COUNT = 10
FINAL_TEST_MAP_COUNT = 20

_LEGACY_MAPS = frozenset(
    (
        DEFAULT_MAP,
        *V4_TRAIN_MAPS.values(),
        *V4_VALIDATION_MAPS.values(),
        *V4_TEST_MAPS.values(),
    )
)


def _split_region(
    grid: list[list[str]],
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    depth: int,
) -> None:
    """Recursively divide a room while leaving a doorway in each partition."""
    x0, x1, y0, y1 = bounds
    orientations: list[str] = []
    if x1 - x0 >= 4:
        orientations.append("vertical")
    if y1 - y0 >= 4:
        orientations.append("horizontal")
    if depth <= 0 or not orientations:
        return

    orientation = rng.choice(orientations)
    if orientation == "vertical":
        wall_x = rng.randint(x0 + 2, x1 - 2)
        for y in range(y0, y1 + 1):
            grid[y][wall_x] = "#"
        door_y = rng.randint(y0, y1)
        grid[door_y][wall_x] = "."
        if y1 - y0 >= 4 and rng.random() < 0.35:
            second_door_y = rng.choice([y for y in range(y0, y1 + 1) if y != door_y])
            grid[second_door_y][wall_x] = "."
        _split_region(grid, rng, (x0, wall_x - 1, y0, y1), depth - 1)
        _split_region(grid, rng, (wall_x + 1, x1, y0, y1), depth - 1)
    else:
        wall_y = rng.randint(y0 + 2, y1 - 2)
        for x in range(x0, x1 + 1):
            grid[wall_y][x] = "#"
        door_x = rng.randint(x0, x1)
        grid[wall_y][door_x] = "."
        if x1 - x0 >= 4 and rng.random() < 0.35:
            second_door_x = rng.choice([x for x in range(x0, x1 + 1) if x != door_x])
            grid[wall_y][second_door_x] = "."
        _split_region(grid, rng, (x0, x1, y0, wall_y - 1), depth - 1)
        _split_region(grid, rng, (x0, x1, wall_y + 1, y1), depth - 1)


def generate_map(seed: int) -> MapRows:
    """Generate a connected room-like 10x10 map deterministically from ``seed``."""
    rng = random.Random(seed)
    grid = [["#" for _ in range(10)] for _ in range(10)]
    for y in range(1, 9):
        for x in range(1, 9):
            grid[y][x] = "."
    _split_region(grid, rng, (1, 8, 1, 8), rng.randint(2, 4))
    return tuple("".join(row) for row in grid)


def _make_split(
    prefix: str,
    seed_range: Iterable[int],
    count: int,
    used_maps: set[MapRows],
) -> dict[str, MapRows]:
    maps: dict[str, MapRows] = {}
    for seed in seed_range:
        rows = generate_map(seed)
        if rows in _LEGACY_MAPS or rows in used_maps:
            continue
        try:
            HideSeekV3Env._validate_map(rows)
        except ValueError:
            continue
        maps[f"{prefix}{len(maps) + 1:03d}"] = rows
        used_maps.add(rows)
        if len(maps) == count:
            return maps
    raise RuntimeError(f"seed range for {prefix} produced only {len(maps)} of {count} maps")


_used_maps: set[MapRows] = set(_LEGACY_MAPS)
TRAIN_MAPS = _make_split("V5T", TRAIN_SEED_RANGE, TRAIN_MAP_COUNT, _used_maps)
VALIDATION_MAPS = _make_split(
    "V5E", VALIDATION_SEED_RANGE, VALIDATION_MAP_COUNT, _used_maps
)
FINAL_TEST_MAPS = _make_split(
    "V5F", FINAL_TEST_SEED_RANGE, FINAL_TEST_MAP_COUNT, _used_maps
)
# Reserved final maps. Keep this alias out of training and validation imports.
TEST_MAPS = FINAL_TEST_MAPS
