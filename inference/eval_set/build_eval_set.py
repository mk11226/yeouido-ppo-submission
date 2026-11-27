"""활성 출발·도착 42쌍을 세 시간대에서 검증하는 126문제 시험지를 만든다.

    python -m inference.eval_set.build_eval_set
    python -m inference.eval_set.build_eval_set --random 30 --seed 26
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # 저장소 최상위
import argparse
import json
import random

from pathlib import Path

from yeouido_cross.build_routes import build as build_routes

OUT = Path(__file__).resolve().parent
LEVELS = ("night", "day", "rush")
SNAPSHOTS = (600, 1200, 1800, 2400)
SUMO_SEED_BASE = 1000


def build():
    _, routes = build_routes()
    problems = []
    for pair_index, route in enumerate(routes):
        for level_index, level in enumerate(LEVELS):
            identifier = len(problems)
            problems.append(dict(
                id=identifier, route_id=route["id"], level=level,
                origin=route["origin"], destination=route["destination"],
                origin_at=[route["edges"][0], route["origin_pos"]],
                destination_at=[route["edges"][-1], route["dest_pos"]],
                snapshot_time=SNAPSHOTS[(pair_index + level_index) % len(SNAPSHOTS)],
                sumo_seed=SUMO_SEED_BASE + identifier,
                category="게이트" if any(route[key].startswith("게이트")
                                    for key in ("origin", "destination")) else route["direction"],
                bridges=route["bridge"].split("+") if route["bridge"] else [],
                route_m=route["length_m"],
            ))
    return problems


def build_random(n, seed):
    """세 시간대에서 고르게 뽑는 재현 가능한 무작위 시험지."""
    rng = random.Random(seed)
    full = build()
    problems = []
    # 시간대별로 문제를 교차 배치해 병렬 실행 시 출근 문제가 몰리지 않게 한다.
    pools = {level: [problem for problem in full if problem["level"] == level]
             for level in LEVELS}
    for pool in pools.values():
        rng.shuffle(pool)
    for identifier in range(n):
        level = LEVELS[identifier % len(LEVELS)]
        pool = pools[level]
        offset = identifier // len(LEVELS)
        problem = pool[offset] if offset < len(pool) else rng.choice(pool)
        problems.append({**problem, "id": identifier,
                         "sumo_seed": SUMO_SEED_BASE + seed * 10_000 + identifier})
    return problems


def check(problems):
    """126문제 모두 env.reset 이 되는지 (경로 있음, 1 km 이상, 출발 지점이 막히지 않음)."""
    import os
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root))
    os.chdir(root)
    from env.mdp_config import ACTION, OBSERVATION, REWARD, SIMULATION
    from env.road_config import EGO, ROAD, TRAFFIC
    from env.yeouido_env import make_env
    env = make_env(cfg_path=None, road=ROAD, ego=EGO, mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
                   action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=False)
    missing = [n for p in problems for n in (p["origin"], p["destination"]) if n not in env.places]
    if missing:
        raise RuntimeError(f"env 의 출발·도착 후보에 없는 이름: {sorted(set(missing))} "
                           "(env/road_config.py places 가 landmarks+gates 인지 확인)")
    failed = []
    for p in problems:
        keys = ("level", "origin", "destination", "origin_at", "destination_at", "snapshot_time", "sumo_seed")
        try:
            _, info = env.reset(options={k: p[k] for k in keys})
            p["route_m"] = round(info["route_length"])      # env(findRoute) 기준 길이로 갱신
        except RuntimeError as e:
            failed.append((p["id"], str(e)))
    env.close()
    return failed


def summary(problems):
    from collections import Counter
    pair_count = len({(p["origin"], p["destination"]) for p in problems})
    print(f"[eval_set] {len(problems)}문제 / {pair_count}쌍")
    for title, key in (("종류", "category"), ("시간대", "level"), ("워밍업 시점", "snapshot_time")):
        print(f"  {title}: " + ", ".join(f"{k} {v}" for k, v in Counter(p[key] for p in problems).items()))
    br = Counter(b for p in problems for b in p["bridges"])
    print("  지나는 샛강 연결로: " + ", ".join(f"{k} {v}" for k, v in br.items()))
    ends = Counter(n for p in {(p["origin"], p["destination"]): p for p in problems}.values() for n in (p["origin"], p["destination"]))
    print("  장소별 등장 (쌍 기준): " + ", ".join(f"{k} {v}" for k, v in sorted(ends.items())))
    print(f"  경로 길이: {min(p['route_m'] for p in problems)}~{max(p['route_m'] for p in problems)} m")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="126문제 모두 env.reset 이 되는지 확인")
    ap.add_argument("--random", type=int, metavar="N",
                    help="고정 126문제 대신 무작위 N문제 (→ inference/eval_set/random_<N>_endpoints7_s<seed>.json)")
    ap.add_argument("--seed", type=int, default=1, help="--random 의 난수 시드")
    args = ap.parse_args(argv)
    if args.random:
        problems = build_random(args.random, args.seed)
        path = OUT / f"random_{args.random}_endpoints7_s{args.seed}.json"
        path.write_text("[\n" + ",\n".join(json.dumps(p, ensure_ascii=False) for p in problems) + "\n]\n")
        summary(problems)
        print(f"→ {path}")
        return
    problems = build()
    if args.check:
        failed = check(problems)
        if failed:
            for pid, msg in failed:
                print(f"   ! {pid}: {msg}")
            raise SystemExit(f"{len(failed)}문제가 reset 되지 않습니다 (파일을 쓰지 않음)")
        print("[eval_set] 126문제 모두 reset 확인")
    OUT.mkdir(exist_ok=True)
    path = OUT / "eval_set_endpoints7.json"
    path.write_text("[\n" + ",\n".join(json.dumps(p, ensure_ascii=False) for p in problems) + "\n]\n")
    summary(problems)
    print(f"→ {path}")


if __name__ == "__main__":
    main()
