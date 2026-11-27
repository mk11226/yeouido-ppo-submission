# 여의도 자율주행 PPO

제출 모델은 1회차 602,112스텝에서 선정한 PPO 정책이다. 입력은 64차원이다. 일반 차량은 심야 462대, 주간 3,338대, 출근 4,417대를 목표로 유지한다. 일반 차량의 초기 경로 설정은 최대 10개 도로 구간이며, 자율주행 차량 주변에서는 경로 종료 전에 연장하고 신규 차량 보충을 제한한다. 정체 완화용 차량 제거는 꺼져 있다. PPO 주행에는 차선 변경 안전 게이트를 적용한다.

출발·도착 번호는 1 국회의사당, 2 IFC 서울, 3 63빌딩, 4 여의도공원, 5 서강대교, 6 마포대교, 7 원효대교다. 방향별 42개 경로를 지원한다.

## 실행

Python 3.12 환경에서 `python -m pip install -r requirements.txt`로 의존성을 설치한다. GUI에는 디스플레이와 Tkinter가 필요하다. 다음 명령은 프로젝트 루트에서 실행한다.

```bash
# 전체 지도
env -u LIBSUMO_AS_TRACI python -m inference.fixed_population_panel --level rush --delay 50

# 최종 PPO 주행
env -u LIBSUMO_AS_TRACI python -m inference.drive_fixed_population_ppo \
  --origin 1 --destination 5 --level rush --delay 50 --hold

# 7개 지점 전체 126회 재검증
LIBSUMO_AS_TRACI=quiet YEOUIDO_TRAFFIC_MODE=fixed_population \
  YEOUIDO_EPISODE_MAX_STEPS=7200 python -m inference.test results/final_ppo/model.pt \
  --no-assist --lane-change-gate --nogui --workers 12 \
  --eval-set inference/eval_set/eval_set_endpoints7.json \
  --out results/final_ppo/revalidation_126.json
```

## 결과와 데이터

- `results/final_ppo/model.pt`는 최종 PPO 가중치다. `environment.json`에 검증 당시의 설정과 모델 해시를 기록했다.
- `validation_selected_126.csv`는 기존 모델의 검증 기록에서 최종 운행 범위에 해당하는 42개 방향별 경로를 심야·주간·출근으로 나누어 집계한 결과다. 완주 100/126회(79.4%), 충돌 1/126회(0.8%)였다. 이 기록은 출발·도착 후보 선정에도 사용했다. 새 시드로 검증하려면 위 재검증 명령을 사용한다.
- `training/data/expert_clean.npz`는 BC에 실제 사용한 364개 전문가 주행, 290,469개 상태·행동 샘플이다. 기존의 짧은 일반 차량 경로 환경에서 수집했다. `observation_schema.csv`와 `npz_fields.json`에서 열 순서와 각 배열의 형식을 확인할 수 있다.
