"""New object-layout partitions; final maps are never used for model selection."""

from __future__ import annotations

import random
import hashlib
import json

from env.hide_seek_v3_env import DEFAULT_MAP
from env.hide_seek_v7_env import Cell, ObjectLayout
from env.v4_maps import TEST_MAPS as V4_TEST_MAPS
from env.v4_maps import TRAIN_MAPS as V4_TRAIN_MAPS
from env.v4_maps import VALIDATION_MAPS as V4_VALIDATION_MAPS
from env.v5_maps import FINAL_TEST_MAPS as V5_FINAL_TEST_MAPS
from env.v5_maps import TRAIN_MAPS as V5_TRAIN_MAPS
from env.v5_maps import VALIDATION_MAPS as V5_VALIDATION_MAPS
from env.v5_maps import generate_map


_PREVIOUS_MAPS = set((
    DEFAULT_MAP,
    *V4_TRAIN_MAPS.values(), *V4_VALIDATION_MAPS.values(), *V4_TEST_MAPS.values(),
    *V5_TRAIN_MAPS.values(), *V5_VALIDATION_MAPS.values(), *V5_FINAL_TEST_MAPS.values(),
))


def make_layout(rows: tuple[str, ...], seed: int) -> ObjectLayout:
    """Place one pushable doorway block and one usable ramp on a room map."""
    size = len(rows)
    floors = {(x, y) for y, row in enumerate(rows) for x, cell in enumerate(row) if cell == "."}
    walls = {(x, y) for y, row in enumerate(rows) for x, cell in enumerate(row) if cell == "#"}
    directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
    ramp_candidates: list[tuple[Cell, Cell, Cell]] = []
    for wall in walls:
        for dx, dy in directions:
            ramp = (wall[0] - dx, wall[1] - dy)
            landing = (wall[0] + dx, wall[1] + dy)
            if ramp in floors and landing in floors:
                ramp_candidates.append((ramp, wall, landing))
    if not ramp_candidates:
        raise ValueError("map has no wall that can hold a ramp")
    rng = random.Random(seed)
    rng.shuffle(ramp_candidates)
    for ramp, low_wall, landing in ramp_candidates:
        block_candidates: list[Cell] = []
        for block in floors - {ramp, landing}:
            pushable = any(
                (block[0] - dx, block[1] - dy) in floors
                and (block[0] + dx, block[1] + dy) in floors
                for dx, dy in ((1, 0), (0, 1))
            )
            if pushable:
                block_candidates.append(block)
        if block_candidates:
            # Prefer doorways. A block in a narrow doorway changes the route.
            block_candidates.sort(key=lambda cell: (
                sum((cell[0] + dx, cell[1] + dy) in floors for dx, dy in directions),
                cell[1], cell[0],
            ))
            narrow = [
                cell for cell in block_candidates
                if sum((cell[0] + dx, cell[1] + dy) in floors for dx, dy in directions)
                == sum((block_candidates[0][0] + dx, block_candidates[0][1] + dy) in floors
                       for dx, dy in directions)
            ]
            block = rng.choice(narrow)
            layout = ObjectLayout(rows, block=block, ramp=ramp, low_wall=low_wall)
            layout.validate()
            return layout
    raise ValueError("map has no valid block position")


def _partition(prefix: str, seed_start: int, count: int, used: set[tuple[str, ...]]) -> dict[str, ObjectLayout]:
    result: dict[str, ObjectLayout] = {}
    for seed in range(seed_start, seed_start + 20_000):
        rows = generate_map(seed)
        if rows in used:
            continue
        try:
            layout = make_layout(rows, seed)
        except ValueError:
            continue
        used.add(rows)
        result[f"{prefix}{len(result) + 1:03d}"] = layout
        if len(result) == count:
            return result
    raise RuntimeError(f"Could not generate {count} {prefix} maps")


_used = set(_PREVIOUS_MAPS)
TRAIN_LAYOUTS = _partition("V7T", 1_000_000, 100, _used)
VALIDATION_LAYOUTS = _partition("V7E", 1_100_000, 10, _used)
FINAL_LAYOUTS = _partition("V7F", 1_200_000, 20, _used)


def layout_hash(layouts: dict[str, ObjectLayout]) -> str:
    payload = [
        (name, layout.map_rows, layout.block, layout.ramp, layout.low_wall)
        for name, layout in layouts.items()
    ]
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
