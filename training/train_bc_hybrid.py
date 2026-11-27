"""PPO와 같은 신경망(HybridActorCritic)으로 하는 Behavior Cloning — PPO 미세조정의 출발점.

    python -m training.train_bc_hybrid --data training/data/expert_dart600.npz
    python -m inference.test results/bc_hybrid_…/model.pt --eval-set inference/eval_set/random_60_s1.json --workers 8

train_bc.py(BCPolicy: 두 행동을 모두 MSE 회귀)와 다른 점:
  - 신경망이 train.py HPARAMS 의 PPO 정책과 똑같다 (256×256 relu, 가감속 μ + 차로 3방향 로짓).
    저장 파일이 PPO state_dict 형식이라 test.py 가 PPO로 읽고, train.py 가 이어서 미세조정할 수 있다.
  - 가감속: tanh(μ) 와 선생님 가감속의 MSE.
  - 차로: 선생님 명령(-1/0/+1)을 3클래스 교차 엔트로피로 (× --lane-weight).
    "유지"가 대부분이라 클래스 빈도의 -0.5 제곱으로 가중한다 (드문 변경도 배우도록).
  - 관측 정규화 없음 (PPO 정책도 원래 관측을 그대로 받는다. 관측은 대부분 [-1, 1]).
  - batch 512, lr 1e-3 cosine 감쇠, 검증 가감속 손실(--select)이 --patience epoch 동안 안 좋아지면 멈춤.
  - 장치: cuda > mps(Apple GPU) > cpu 자동 선택.
저장 시 가감속 σ(log_std)를 --save-log-std 로 맞춘다 (PPO 미세조정 시작 탐험 폭, 기본 e^-1.6 ≈ 0.2).
가치(critic) head는 학습하지 않는다 → PPO 미세조정 초반에 critic 만 먼저 맞춘다.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # 저장소 최상위 (env, utils 등)
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.networks import HybridActorCritic

BASE = Path(__file__).resolve().parent.parent   # 저장소 최상위


def load(path):
    with np.load(path, allow_pickle=False) as d:
        obs = d["observation"].astype(np.float32)
        act = d["action_raw"].astype(np.float32)
        ep = d["episode"].astype(np.int64)
    if not (np.isfinite(obs).all() and np.isfinite(act).all()):
        raise ValueError("관측/행동에 NaN 또는 Inf가 있습니다.")
    return obs, act, ep


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sample_lane_indices(labels, train_indices, size, change_fraction):
    """Balance the lane loss without changing the acceleration-data distribution."""
    changes = train_indices[labels[train_indices] != 1]
    keeps = train_indices[labels[train_indices] == 1]
    if not len(changes) or not len(keeps):
        raise ValueError("차로 균형 학습에는 변경·유지 시범이 모두 필요합니다.")
    n_change = max(1, min(size - 1, round(size * change_fraction)))
    change_idx = changes[torch.randint(len(changes), (n_change,), device=labels.device)]
    keep_idx = keeps[torch.randint(len(keeps), (size - n_change,), device=labels.device)]
    return torch.cat((change_idx, keep_idx))


def acceleration_loss(prediction,target,brake_weight=1.0):
    """Prioritize braking demonstrations without changing the driving controls."""
    weights=torch.where(target<-.2,brake_weight,1.0)
    return ((prediction-target).square()*weights).sum()/weights.sum()


def sample_source_indices(pools,size,fractions,labels=None,change_fraction=None):
    """Draw a fixed source mixture without admitting validation indices."""
    weights=np.asarray(fractions,dtype=float)
    if len(pools)!=len(weights) or not np.isfinite(weights).all() or (weights<0).any() or not np.isclose(weights.sum(),1):
        raise ValueError('Source fractions must be finite, nonnegative and sum to one')
    expected=weights*size;counts=np.floor(expected).astype(int)
    for index in np.argsort(-(expected-counts))[:size-int(counts.sum())]:counts[index]+=1
    samples=[]
    for pool,count in zip(pools,counts):
        if count==0:continue
        if len(pool)==0:raise ValueError('A sampled source has no training examples')
        if labels is not None and change_fraction is not None and count>1:
            lane_labels=labels[pool]
            if bool((lane_labels!=1).any()) and bool((lane_labels==1).any()):
                samples.append(sample_lane_indices(labels,pool,count,change_fraction));continue
        samples.append(pool[torch.randint(len(pool),(int(count),),device=pool.device)])
    return torch.cat(samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, nargs="+", help="collect_pomdp_data.py 의 .npz (여러 개면 합친다)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lane-weight", type=float, default=5.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--select", default="accel", choices=["accel", "total", "balanced"],
                    help="모델을 고르는 검증 손실. accel: 가감속 MSE (차로 변경 라벨이 수십 개뿐이라 "
                         "차로 손실은 epoch마다 크게 흔들려 멈출 시점을 망친다), total: 가감속 + 차로×가중치")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--save-log-std", type=float, default=-1.6)
    ap.add_argument("--init", default=None, help="이어서 학습할 model.pt (DAgger 반복용)")
    ap.add_argument("--lane-change-fraction", type=float, default=None,
                    help="차로 손실 배치의 변경 시범 비율 (가감속 배치는 원래 분포)")
    ap.add_argument("--lane-only", action="store_true", help="공유 신경망·가감속을 고정하고 차로 출력만 학습")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--save-candidates", action="store_true",
                    help="가감속·차선 균형·전체 손실 기준 최적 체크포인트를 각각 보존")
    ap.add_argument("--brake-weight", type=float, default=1.0,
                    help="정답 감속 raw < -0.2 시범의 가감속 손실 가중치 (기본 1)")
    ap.add_argument('--source-fractions',nargs='+',type=float,default=None,
                    help='각 --data 파일이 학습 배치에서 차지할 비율, 예: 0.6 0.4')
    args = ap.parse_args()
    if args.lane_change_fraction is not None and not 0 < args.lane_change_fraction < 1:
        ap.error("--lane-change-fraction은 0과 1 사이여야 합니다.")
    if args.lane_only and not args.init:
        ap.error("--lane-only에는 --init 모델이 필요합니다.")
    if not np.isfinite(args.brake_weight) or not 1<=args.brake_weight<=20:
        ap.error('--brake-weight는 1 이상 20 이하의 유한수여야 합니다.')
    if args.source_fractions is not None:
        fractions=np.asarray(args.source_fractions)
        if len(fractions)!=len(args.data) or not np.isfinite(fractions).all() or (fractions<0).any() or not np.isclose(fractions.sum(),1):
            ap.error('--source-fractions는 데이터 파일 수와 같고 합이 1이어야 합니다.')

    from training.train import HPARAMS
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)

    parts = [load(p) for p in args.data]
    offset, obs, act, ep, source_ids = 0, [], [], [], []
    for source_index,(o, a, e) in enumerate(parts):  # 파일마다 에피소드 번호가 겹치지 않게
        obs.append(o), act.append(a), ep.append(e + offset)
        source_ids.append(np.full(len(o),source_index,dtype=np.int16))
        offset = int(ep[-1].max()) + 1
    obs, act, ep = np.concatenate(obs), np.concatenate(act), np.concatenate(ep)
    source_ids=np.concatenate(source_ids)
    lane_idx = (1 - np.round(act[:, 1])).astype(np.int64).clip(0, 2)   # +1→0(왼), 0→1(유지), -1→2(오른)

    eps = rng.permutation(np.unique(ep))
    if args.source_fractions is None:
        n_val = max(1, int(round(len(eps) * args.val_fraction)))
        val_mask = np.isin(ep, eps[:n_val])
    else:
        val_mask=np.zeros(len(ep),dtype=bool)
        for group in range(len(parts)):
            group_eps=np.random.default_rng(args.seed+group).permutation(np.unique(ep[source_ids==group]))
            if len(group_eps)<2:raise ValueError('Each source needs at least two episodes for a held-out split')
            n_val=max(1,min(len(group_eps)-1,round(len(group_eps)*args.val_fraction)))
            val_mask|=np.isin(ep,group_eps[:n_val])
    tr, va = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)

    counts = np.bincount(lane_idx[tr], minlength=3).astype(np.float64)
    cls_w = (counts / counts.sum() + 1e-6) ** -0.5
    # 선생님의 차로 변경은 아주 드물다 (경로상 변경은 SUMO가 직접 한다: 기존 데이터 8만 스텝 중 17번).
    # 가중치를 끝까지 키우면 쓸데없는 변경을 배우므로 "유지"의 10배까지로 자른다.
    cls_w = np.minimum(cls_w, 10.0 * cls_w[1])
    cls_w = cls_w / (cls_w * counts / counts.sum()).sum()      # 평균 가중치 1
    if args.lane_change_fraction is not None:
        cls_w = np.ones(3)  # 샘플링과 클래스 가중치를 중복 적용하지 않는다.

    policy = HybridActorCritic(obs.shape[1], hidden_sizes=HPARAMS["hidden_sizes"],
                               activation=HPARAMS["activation"],
                               init_log_std=HPARAMS["init_log_std"],
                               lane_keep_bias=HPARAMS["lane_keep_bias"]).to(device)
    if args.init:
        policy.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
    if args.lane_only:
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(name.startswith("lane_head"))
    actor = [p for n, p in policy.named_parameters() if not n.startswith(("v_head", "log_std"))]
    actor = [p for p in actor if p.requires_grad]
    opt = torch.optim.Adam(actor, lr=args.lr)
    steps_per_epoch = int(np.ceil(len(tr) / args.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch,
                                                       eta_min=args.lr * 0.05)
    w_t = torch.as_tensor(cls_w, dtype=torch.float32, device=device)

    # 데이터 전체를 장치에 올려 둔다 (100만 건 × 42 ≈ 170 MB). 미니배치 복사 왕복이 없어진다.
    O = torch.as_tensor(obs, device=device)
    A = torch.as_tensor(act[:, 0:1], device=device)
    L = torch.as_tensor(lane_idx, device=device)
    tr_t, va_t = torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device)
    source_pools=[torch.as_tensor(tr[source_ids[tr]==group],device=device) for group in range(len(parts))]

    def batch_loss(idx):
        mu, _, logits, _ = policy(O[idx])
        accel_loss = acceleration_loss(mu, A[idx], args.brake_weight)
        lane_loss = F.cross_entropy(logits, L[idx], weight=w_t)
        return accel_loss, lane_loss, logits

    out = Path(args.out_dir) if args.out_dir else BASE / "results" / datetime.now().strftime("bc_hybrid_%m%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), device=device, samples=len(obs), train=len(tr), val=len(va),
                  source_train_samples=[len(pool) for pool in source_pools],
                  observation_dim=int(obs.shape[1]),
                  unassisted=_os.environ.get("YEOUIDO_UNASSISTED") == "1",
                  episodes=len(eps), lane_counts=counts.tolist(), lane_class_weight=cls_w.tolist(),
                  hidden_sizes=list(HPARAMS["hidden_sizes"]), activation=HPARAMS["activation"])
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"BC(hybrid) 장치 {device} | 샘플 {len(obs):,} (학습 {len(tr):,} / 검증 {len(va):,}), "
          f"에피소드 {len(eps)} | 차로 라벨 왼/유지/오 {counts.astype(int).tolist()} | → {out}", flush=True)

    best, bad = float("inf"), 0
    candidate_scores = {name:float('inf') for name in ('accel','balanced','total')}
    log = open(out / "training_log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["epoch", "train_accel", "train_lane", "val_accel", "val_lane", "val_total",
                "val_lane_acc", "val_change_recall", "lr", "val_lane_macro"])
    for epoch in range(1, args.epochs + 1):
        policy.train()
        perm = tr_t[torch.randperm(len(tr_t), device=device)] if args.source_fractions is None else tr_t
        ta = tl = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            if args.source_fractions is not None:
                idx=sample_source_indices(source_pools,len(idx),args.source_fractions)
            al, ll, _ = batch_loss(idx)
            if args.lane_change_fraction is not None:
                lane_idx_batch = (sample_lane_indices(L,tr_t,len(idx),args.lane_change_fraction)
                    if args.source_fractions is None else sample_source_indices(source_pools,len(idx),
                        args.source_fractions,L,args.lane_change_fraction))
                _, ll, _ = batch_loss(lane_idx_batch)
            loss = args.lane_weight * ll if args.lane_only else al + args.lane_weight * ll
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor, 1.0)
            opt.step()
            sched.step()
            ta += al.item() * len(idx)
            tl += ll.item() * len(idx)
        ta, tl = ta / len(tr), tl / len(tr)

        policy.eval()
        va_a = va_l = 0.0
        correct = change_hit = change_n = 0
        class_loss = torch.zeros(3, device=device)
        class_count = torch.zeros(3, device=device)
        with torch.no_grad():
            for i in range(0, len(va_t), 8192):
                idx = va_t[i:i + 8192]
                al, ll, logits = batch_loss(idx)
                va_a += al.item() * len(idx)
                va_l += ll.item() * len(idx)
                pred = logits.argmax(-1)
                correct += (pred == L[idx]).sum().item()
                ch = L[idx] != 1
                change_n += ch.sum().item()
                change_hit += ((pred == L[idx]) & ch).sum().item()
                per_sample = F.cross_entropy(logits, L[idx], reduction="none")
                for category in range(3):
                    mask = L[idx] == category
                    class_loss[category] += per_sample[mask].sum()
                    class_count[category] += mask.sum()
        va_a, va_l = va_a / len(va), va_l / len(va)
        total = va_a + args.lane_weight * va_l
        recall = change_hit / max(change_n, 1)
        present = class_count > 0
        macro = float((class_loss[present] / class_count[present]).mean().item())
        lr = sched.get_last_lr()[0]
        w.writerow([epoch, f"{ta:.5f}", f"{tl:.5f}", f"{va_a:.5f}", f"{va_l:.5f}", f"{total:.5f}",
                    f"{correct / len(va):.4f}", f"{recall:.4f}", f"{lr:.2e}", f"{macro:.5f}"])
        log.flush()
        mark = ""
        score = (va_a if args.select == "accel" else
                 macro if args.select == "balanced" and args.lane_only else
                 va_a + args.lane_weight * macro if args.select == "balanced" else total)
        if args.save_candidates:
            scores=dict(accel=va_a,balanced=va_a+args.lane_weight*macro,total=total)
            for name,candidate_score in scores.items():
                if candidate_score<candidate_scores[name]-1e-5:
                    candidate_scores[name]=candidate_score
                    folder=out/'candidates'/name
                    folder.mkdir(parents=True,exist_ok=True)
                    state={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
                    if not args.lane_only:
                        state['log_std'].fill_(args.save_log_std)
                    torch.save(state,folder/'model.pt')
                    (folder/'selection.json').write_text(json.dumps(dict(epoch=epoch,
                        criterion=name,score=candidate_score,val_accel=va_a,
                        val_lane_macro=macro,val_lane_change_recall=recall),indent=2))
        if score < best - 1e-5:
            best, bad, mark = score, 0, " *"
            with torch.no_grad():
                if not args.lane_only:
                    policy.log_std.fill_(args.save_log_std)
            torch.save({k: v.cpu() for k, v in policy.state_dict().items()}, out / "model.pt")
        else:
            bad += 1
        print(f"epoch {epoch:3d} | 학습 가감속 {ta:.4f} 차로 {tl:.4f} | 검증 가감속 {va_a:.4f} 차로 {va_l:.4f} "
              f"합 {total:.4f} 매크로 {macro:.4f} | 차로 정확도 {correct / len(va):.3f} 변경 재현율 {recall:.3f} | lr {lr:.1e}{mark}",
              flush=True)
        if bad >= args.patience:
            print(f"검증 손실이 {args.patience} epoch 동안 좋아지지 않아 멈춤")
            break
    log.close()
    print(f"최고 검증 손실 {best:.4f} → {out / 'model.pt'}")


if __name__ == "__main__":
    main()
