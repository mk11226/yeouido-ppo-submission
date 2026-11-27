"""Open the final PPO in the fixed-population SUMO GUI."""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--max-steps", type=int, default=7200)
    args, rest = parser.parse_known_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if os.environ.get("LIBSUMO_AS_TRACI"):
        parser.error("GUI 실행 시 LIBSUMO_AS_TRACI를 해제하세요.")
    if any(arg in rest for arg in ("--reroute", "--no-assist", "--change-dest", "--teacher-assisted")):
        parser.error("이 실행 명령은 최종 PPO의 주행 설정을 사용합니다.")
    os.environ.update(
        YEOUIDO_TRAFFIC_MODE="fixed_population", YEOUIDO_UNASSISTED="1",
        YEOUIDO_LANE_CHANGE_GATE="1", YEOUIDO_CONGESTION_RELIEF="0",
        YEOUIDO_MIN_TRIP="0", YEOUIDO_EPISODE_MAX_STEPS=str(args.max_steps),
    )
    from yeouido.config import sumo_home
    os.environ["SUMO_HOME"] = str(sumo_home())
    from env.fixed_population import prepare
    from inference import drive_model

    if "--help" not in rest and "-h" not in rest:
        prepare()
    sys.argv = [sys.argv[0], *rest]
    drive_model.main()


if __name__ == "__main__":
    main()
