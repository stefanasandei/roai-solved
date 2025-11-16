import os
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def load_and_clean_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    df.rename(
        columns={c: re.sub(r"[^A-Za-z0-9_]+", "_", c.strip()) for c in df.columns},
        inplace=True,
    )

    for col in [
        "Arrivals",
        "Occupancy",
        "Wind_X_m_s_",
        "Wind_Y_m_s_",
        "Insects_Delta_g_m_",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def infer_grid_size(df: pd.DataFrame) -> int:
    day = df["Day"].mode().iat[0]
    n_bops = df[df["Day"] == day]["BOP"].nunique()
    side = int(round(np.sqrt(n_bops)))
    if side * side != n_bops:
        raise ValueError(
            f"Cannot infer square grid from {n_bops} BOPs. Expected perfect square."
        )
    return side


def id_to_rc(bop_id: int, side: int) -> Tuple[int, int]:
    bop_id = int(bop_id)
    r = (bop_id - 1) // side
    c = (bop_id - 1) % side
    return r, c


def rc_to_id(r: int, c: int, side: int) -> int:
    return r * side + c + 1


def get_d4_transforms(side: int):
    def I(r, c):
        return r, c

    def R90(r, c):
        return c, side - 1 - r

    def R180(r, c):
        return side - 1 - r, side - 1 - c

    def R270(r, c):
        return side - 1 - c, r

    def FV(r, c):  # vertical mirror (left-right)
        return r, side - 1 - c

    def FH(r, c):  # horizontal mirror (top-bottom)
        return side - 1 - r, c

    def FD(r, c):  # main diagonal
        return c, r

    def FAD(r, c):  # anti-diagonal
        return side - 1 - c, side - 1 - r

    ops = {
        "I": I,
        "R90": R90,
        "R180": R180,
        "R270": R270,
        "FV": FV,
        "FH": FH,
        "FD": FD,
        "FAD": FAD,
    }

    def make_map(fn):
        return lambda bop_id: rc_to_id(*fn(*id_to_rc(bop_id, side)), side)

    return {name: make_map(fn) for name, fn in ops.items()}


def top50_by_day(df: pd.DataFrame) -> Dict[int, List[int]]:
    res: Dict[int, List[int]] = {}
    for day, g in df.groupby("Day"):
        # highest Arrivals first
        top = g.nlargest(50, "Arrivals")["BOP"].astype(int).tolist()
        res[int(day)] = top
    return res


def apply_transform(ids: Iterable[int], f) -> List[int]:
    return [int(f(int(b))) for b in ids]


def sets_equal(a: Iterable[int], b: Iterable[int]) -> bool:
    return set(a) == set(b)


def evaluate_mapping_window(
    s6: Dict[int, List[int]],
    s5: Dict[int, List[int]],
    days6: List[int],
    trans_fn,
    map_day5,
) -> Tuple[int, List[bool]]:
    # returns number of exact matches and per-day booleans
    matches = []
    for d6 in days6:
        d5 = map_day5(d6)
        if d5 not in s5:
            matches.append(False)
            continue
        transformed = apply_transform(s5[d5], trans_fn)
        matches.append(sets_equal(s6[d6], transformed))
    return sum(matches), matches


def find_best_mapping(
    s6: Dict[int, List[int]],
    s5: Dict[int, List[int]],
    transforms: Dict[str, callable],
    days6: List[int],
) -> Tuple[str, str, callable]:
    # Try two families:
    # 1) Reversed time: d5 = C - d6, with C chosen so all mapped days within [1..730]
    #    Most plausible C is 1491 -> maps 761..790 -> 730..701
    # 2) Year offset forward: d5 = d6 + K, K so mapped days within [1..730]
    #    Most plausible K is -730 -> 761..790 -> 31..60
    candidates = []

    # build allowed day5 range
    all_days5 = sorted(s5.keys())
    min5, max5 = all_days5[0], all_days5[-1]

    # reversed family
    # compute C range so that for all d6 in days6, min5 <= C - d6 <= max5
    C_min = min5 + max(days6)
    C_max = max5 + min(days6)
    for C in range(C_min, C_max + 1):

        def map_rev(d6, C=C):
            return C - d6

        for tname, tf in transforms.items():
            score, mask = evaluate_mapping_window(s6, s5, days6, tf, map_rev)
            candidates.append(("reversed", f"C={C}", tname, tf, score, mask))

    # forward offset family
    # K range so that for all d6 in days6, min5 <= d6 + K <= max5
    K_min = min5 - min(days6)
    K_max = max5 - max(days6)
    for K in range(K_min, K_max + 1):

        def map_fw(d6, K=K):
            return d6 + K

        for tname, tf in transforms.items():
            score, mask = evaluate_mapping_window(s6, s5, days6, tf, map_fw)
            candidates.append(("forward", f"K={K}", tname, tf, score, mask))

    # Pick best
    candidates.sort(key=lambda x: x[4], reverse=True)
    best = candidates[0]
    family, param, tname, tf, score, mask = best

    # Prefer exact 30/30 match if available
    for cand in candidates:
        if cand[4] == len(days6):
            family, param, tname, tf, score, mask = cand
            break

    return f"{family}:{param}", tname, tf


def main():
    DATA_DIR = "level_6"
    f6 = os.path.join(DATA_DIR, "level_6.in")
    f5 = os.path.join(DATA_DIR, "all_data_from_level_5.in")

    if not os.path.exists(f6) or not os.path.exists(f5):
        raise FileNotFoundError("Expected Level 6/level_6.in and level_5.in")

    df6 = load_and_clean_csv(f6)
    df5 = load_and_clean_csv(f5)

    # sanity: keep only relevant columns
    needed = {"Day", "BOP", "Arrivals"}
    if not needed.issubset(df6.columns) or not needed.issubset(df5.columns):
        missing = needed - set(df6.columns.union(df5.columns))
        raise ValueError(f"Missing required columns: {missing}")

    side = infer_grid_size(df6)
    transforms = get_d4_transforms(side)

    s6 = top50_by_day(df6)
    s5 = top50_by_day(df5)

    days6 = list(range(761, 791))

    # Find mapping that aligns L6 days to some L5 days + grid transform
    where, _, tf = find_best_mapping(s6, s5, transforms, days6)

    # Parse mapping to compute the mapped day for 791
    family, param = where.split(":")
    if family == "reversed":
        C = int(param.split("=")[1])
        map_day = lambda d: C - d
    else:
        K = int(param.split("=")[1])
        map_day = lambda d: d + K

    d5_target = map_day(791)
    if d5_target not in s5:
        # Fallback: most plausible two options
        # 1) reversed with C=1491 -> 791 -> 700
        # 2) forward K=-730 -> 791 -> 61
        if 700 in s5:
            d5_target = 700
            tf = transforms.get("R180", list(transforms.values())[0])
        elif 61 in s5:
            d5_target = 61
            tf = transforms.get("I", list(transforms.values())[0])
        else:
            # Ultimate fallback: repeat day 790
            top50 = s6[790]
            out = pd.DataFrame(
                [{"Day": 791, "Top 50 Arrivals BOPs": " ".join(map(str, top50))}]
            )
            out.to_csv("level_6_submission.csv", index=False)
            return

    top50_l5 = s5[d5_target]
    pred_ids = apply_transform(top50_l5, tf)

    out = pd.DataFrame(
        [{"Day": 791, "Top 50 Arrivals BOPs": " ".join(map(str, pred_ids))}]
    )
    out.to_csv("level_6_submission.csv", index=False)


if __name__ == "__main__":
    main()
