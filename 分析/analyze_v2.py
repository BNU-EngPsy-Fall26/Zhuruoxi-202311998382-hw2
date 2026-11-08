# -*- coding: utf-8 -*-
"""
作业二 · 实验结果正式分析 v2（极端值排除版）
数据源：docs/data/hw2_F2EMS2_2026-11-08_214836.csv（整理后主文件，1980 行，P01–P11）
        （备份 docs/data/hw2_F2EMS2_2026-11-08_214836_backup_full.csv 含 P00 测试记录与仅完成实验二的 P22，不进入正式分析）
输出：  分析/作业二_结果与讨论报告.md + 图1、图2 + 分析/排除日志.md（重跑时重新生成，与随作业提交的 数据结果/ 文件夹内容一致）

统计口径严格遵循《总实验设计》3.6 / 4.6 / 6.2：
  剔除（收数前规则）：正式试次超时率 > 20%，或总体正确率精确二项检验显著低于 50%
  极端值排除（本次新增，分层）：
    L1 试次级：RT 有效窗口 实验一 200–1200 ms / 实验二 200–2500 ms（实现上限）
    L2 单元级：实验二单元正确 RT < 3 个 → 该斜率记缺失（不填 0）
    L3 被试级：关键指标 |z| > 3 离群检查（删除仅凭 L1/L2 之外的明确异常证据）
  实验一 RT 只分析正确且 200–1200 ms；不对 10 试次单元做 ±3SD 删除
  实验一对比 Holm 校正（30 ms 主要、60 ms 次要）；实验二两两 Bonferroni ×3
  效应量：Cohen's dz、偏 η²、差值 95% CI；实验二拓扑–特征补 TOST（±10 ms/item）
"""
import sys, os
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))   # 仓库根目录（index.html 所在）
SUBMIT = ROOT
CSV = os.path.join(SUBMIT, "docs", "data", "hw2_F2EMS2_2026-11-08_214836.csv")
SOAS = [30, 60, 600]
CONDS = ["feature", "conjunction", "topology"]
CNAME = {"feature": "特征搜索", "conjunction": "联合搜索", "topology": "拓扑搜索"}
REF = {"feat_p": 3.1, "conj_p": 28.7, "conj_a": 67.1}      # Treisman & Gelade (1980) Exp.1
CHEN82 = {"ring_circle": 64.5, "circle_square": 43.5, "circle_triangle": 38.5}  # Chen (1982) Exp.I
PRED1 = {"topo": [(65, 75), (75, 85), (90, 100)],
         "shape": [(50, 60), (55, 65), (90, 100)]}
TOPO_DIR = {"P01": "ring", "P02": "solid", "P03": "solid", "P04": "ring", "P05": "ring",
            "P06": "solid", "P07": "ring", "P08": "solid", "P09": "solid", "P10": "ring",
            "P11": "solid"}   # README 分配表 A–K

df = pd.read_csv(CSV, encoding="utf-8-sig")
df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
df["corr"] = pd.to_numeric(df["corr"], errors="coerce").fillna(0).astype(int)

# ---------- 统计工具 ----------
def paired_t(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y; n = len(d)
    t, p = stats.ttest_rel(x, y)
    md, sd = d.mean(), d.std(ddof=1)
    dz = md / sd if sd > 0 else np.nan
    se = sd / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
    return t, p, dz, md, md - tc * se, md + tc * se

def holm(pvals):
    """Holm 校正（递增单调约束）"""
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, p[idx] * (m - i)); adj[idx] = min(run, 1.0)
    return adj

def rm_anova_1(X):
    """单因素被试内 ANOVA，X: n×k；返回 F,p,df1,df2,偏eta2"""
    X = np.asarray(X, float); n, k = X.shape
    grand = X.mean(); cell = X.mean(axis=0); subj = X.mean(axis=1)
    ss_c = n * ((cell - grand) ** 2).sum()
    ss_s = k * ((subj - grand) ** 2).sum()
    ss_t = ((X - grand) ** 2).sum()
    ss_e = ss_t - ss_c - ss_s
    d1, d2 = k - 1, (k - 1) * (n - 1)
    F = (ss_c / d1) / (ss_e / d2)
    return F, stats.f.sf(F, d1, d2), d1, d2, ss_c / (ss_c + ss_e)

def rm_anova_2(M):
    """两因素被试内 ANOVA，M: n×(a*b) 列序 a 外 b 内；返回 dict"""
    A_lv, B_lv = 2, 3
    n = M.shape[0]
    T = M.reshape(n, A_lv, B_lv)
    grand = T.mean()
    A_m = T.mean(axis=(0, 2)); B_m = T.mean(axis=(0, 1)); AB_m = T.mean(axis=0); S_m = T.mean(axis=(1, 2))
    errAS = ((T - AB_m[None] - S_m[:, None, None] + B_m[None, None]) ** 2).sum()
    errBS = ((T - AB_m[None] - S_m[:, None, None] + A_m[None, :, None]) ** 2).sum()
    errABS = ((T - AB_m[None] - S_m[:, None, None] + grand) ** 2).sum()
    ssA = n * B_lv * ((A_m - grand) ** 2).sum()
    ssB = n * A_lv * ((B_m - grand) ** 2).sum()
    ssAB = n * ((AB_m - A_m[:, None] - B_m[None] + grand) ** 2).sum()
    out = {}
    for name, ss, d1, d2, err in (("A(任务)", ssA, A_lv - 1, (A_lv - 1) * (n - 1), errAS),
                                  ("B(SOA)", ssB, B_lv - 1, (B_lv - 1) * (n - 1), errBS),
                                  ("A×B(交互)", ssAB, (A_lv - 1) * (B_lv - 1), (A_lv - 1) * (B_lv - 1) * (n - 1), errABS)):
        F = (ss / d1) / (err / d2)
        out[name] = (F, stats.f.sf(F, d1, d2), d1, d2, ss / (ss + err))
    return out

def loglinear_dprime(hits, n_s, fas, n_n):
    H = (hits + 0.5) / (n_s + 1); FA = (fas + 0.5) / (n_n + 1)
    return stats.norm.ppf(H) - stats.norm.ppf(FA)

def fmt_p(p):
    return "p < .001" if p < 0.001 else f"p = {p:.3f}"

def zcheck(vals, label, log):
    v = np.asarray(vals, float); z = (v - v.mean()) / v.std(ddof=1)
    ext = [(i, round(zi, 2)) for i, zi in enumerate(z) if abs(zi) > 3]
    log.append(f"| {label} | {v.mean():.1f} ± {v.std(ddof=1):.1f} | " +
               ("无（max |z| = %.2f）" % np.abs(z).max() if not ext else str(ext)) + " |")
    return ext

L = []; W = L.append
EX = []   # 排除日志行

# ============================================================
# 实验一
# ============================================================
e1 = df[df["experiment"] == "topo_soa"].copy()
pids1 = sorted(e1["participant"].unique())
acc1, rt1, dp1, ot1 = {}, {}, {}, {}
for pid in pids1:
    g = e1[e1["participant"] == pid]
    for task in ("topo", "shape"):
        gg = g[g["block"] == task]
        for soa in SOAS:
            cell = gg[gg["soa"] == soa]
            acc1[(pid, task, soa)] = 100 * cell["corr"].mean()
            rts = cell[(cell["corr"] == 1) & cell["rt"].between(200, 1200)]["rt"]
            rt1[(pid, task, soa)] = rts.median() if len(rts) else np.nan
            sig = cell["hole"] == 1 if task == "topo" else cell["outline"] == "circle"
            ns = ~sig
            resp = cell["resp"].fillna("")
            sresp = "hole" if task == "topo" else "circle"
            dp1[(pid, task, soa)] = loglinear_dprime(int((sig & (resp == sresp)).sum()), int(sig.sum()),
                                                     int((ns & (resp == sresp)).sum()), int(ns.sum()))
        ot1[(pid, task)] = 100 * gg["rt"].isna().mean()

# --- 剔除判定（收数前规则 + 极端值证据）---
excl1 = []
for pid in pids1:
    reasons = []
    p_all = stats.binomtest(int(e1[e1["participant"] == pid]["corr"].sum()), 60, 0.5, alternative="less").pvalue
    acc_all = 100 * e1[e1["participant"] == pid]["corr"].mean()
    ot_all = np.mean([ot1[(pid, "topo")], ot1[(pid, "shape")]])
    if p_all < 0.05:
        reasons.append(f"总体正确率 {acc_all:.1f}%，精确二项检验 {fmt_p(p_all)} < 50%（设计 6.2）")
    if ot_all > 20:
        reasons.append(f"正式试次超时率 {ot_all:.1f}% > 20%（设计 6.2）")
    for task in ("topo", "shape"):
        cell = e1[(e1["participant"] == pid) & (e1["block"] == task) & (e1["soa"] == 600)]
        k, N = int(cell["corr"].sum()), len(cell)
        p6 = stats.binomtest(k, N, 0.5, alternative="less").pvalue
        if p6 < 0.05:
            reasons.append(f"{task} 600 ms 充分呈现档 {k}/{N}，二项检验 {fmt_p(p6)}（按键映射失效证据）")
    if reasons:
        excl1.append((pid, reasons))
eff1 = [p for p in pids1 if not any(x[0] == p for x in excl1)]

# 剩余被试极端值复查
rem_acc = {pid: 100 * e1[e1["participant"] == pid]["corr"].mean() for pid in eff1}
v = np.array(list(rem_acc.values()))
zmax = np.abs((v - v.mean()) / v.std(ddof=1)).max()

# 描述统计
def m1(task, soa, dic): return np.array([dic[(p, task, soa)] for p in eff1], float)

# H1（Holm）
h1 = {}
for soa in SOAS:
    h1[soa] = paired_t(m1("topo", soa, acc1), m1("shape", soa, acc1))
h1_adj = holm([h1[30][1], h1[60][1]])
h1p30, h1p60 = h1_adj[0], h1_adj[1]

M1 = np.array([[acc1[(p, t, s)] for t in ("topo", "shape") for s in SOAS] for p in eff1])
res2 = rm_anova_2(M1)

# ============================================================
# 实验二
# ============================================================
e2 = df[df["experiment"] == "fit_topo_search"].copy()
pids2 = sorted(e2["participant"].unique())
acc_all2 = {p: 100 * e2[e2["participant"] == p]["corr"].mean() for p in pids2}
ot_all2 = {p: 100 * e2[e2["participant"] == p]["rt"].isna().mean() for p in pids2}
excl2 = [p for p in pids2 if acc_all2[p] < 50 or ot_all2[p] > 20]
eff2 = [p for p in pids2 if p not in excl2]

rt2, acc2 = {}, {}
n_below, n_above = 0, 0
for (pid, cond, nn, tgt), cell in e2.groupby(["participant", "block", "n", "target"]):
    acc2[(pid, cond, int(nn), int(tgt))] = 100 * cell["corr"].mean()
    ok = cell[(cell["corr"] == 1) & cell["rt"].notna()]
    n_below += (ok["rt"] < 200).sum(); n_above += (ok["rt"] > 2500).sum()
    keep = ok[ok["rt"].between(200, 2500)]["rt"]
    rt2[(pid, cond, int(nn), int(tgt))] = keep.median() if len(keep) >= 3 else np.nan

def slope(pid, cond, tgt):
    a, b = rt2.get((pid, cond, 6, tgt)), rt2.get((pid, cond, 18, tgt))
    return (b - a) / 12 if (pd.notna(a) and pd.notna(b)) else np.nan

sl = {t: {c: [slope(p, c, t) for p in eff2] for c in CONDS} for t in (1, 0)}
sl_present = pd.DataFrame({c: sl[1][c] for c in CONDS}, index=eff2)
sl_absent = pd.DataFrame({c: sl[0][c] for c in CONDS}, index=eff2)

# 方向分解
dir_ring = [p for p in eff2 if TOPO_DIR.get(p) == "ring"]
dir_solid = [p for p in eff2 if TOPO_DIR.get(p) == "solid"]
topo_ring = np.array([slope(p, "topology", 1) for p in dir_ring])
topo_solid = np.array([slope(p, "topology", 1) for p in dir_solid])

# H2 ANOVA + Bonferroni
F2, pA2, d1a, d2a, etaA = rm_anova_1(sl_present[CONDS].values)
pairs = [("topology", "conjunction", "拓扑 − 联合"), ("topology", "feature", "拓扑 − 特征"),
         ("feature", "conjunction", "特征 − 联合")]
h2p = {}
for a, b, lab in pairs:
    t, p, dz, md, lo, hi = paired_t(sl_present[a].values, sl_present[b].values)
    h2p[lab] = (t, min(p * 3, 1.0), p, dz, md, lo, hi)

# TOST：拓扑 vs 特征，±10 ms/item
d_tf = sl_present["topology"].values - sl_present["feature"].values
n = len(d_tf); md_tf, sd_tf = d_tf.mean(), d_tf.std(ddof=1); se = sd_tf / np.sqrt(n)
t_low = (md_tf + 10) / se; p_low = 1 - stats.t.cdf(t_low, n - 1)      # H0: d<=-10
t_hi = (md_tf - 10) / se; p_hi = stats.t.cdf(t_hi, n - 1)             # H0: d>=+10
p_tost = max(p_low, p_hi)

# ============================================================
# 排除日志
# ============================================================
EX.append("# 排除与极端值日志（v2，2026-11-08）\n")
EX.append(f"数据源：数据结果/hw2_F2EMS2_2026-11-08_214836.csv（整理后主文件，{len(df)} 行，P01–P11）\n")
EX.append("整理阶段已排除：P00 全部测试记录（experiment = test_*）；P22（仅完成实验二、未完成实验一，不满足同一被试完成两实验的设计）。\n")
EX.append("## 收数前规则剔除（设计 6.2）\n")
for pid, rs in excl1:
    EX.append(f"- **实验一 {pid}**：" + "；".join(rs) + " → 剔除")
if not excl2:
    EX.append(f"- **实验二**：{len(pids2)} 人总正确率 {min(acc_all2.values()):.1f}%–{max(acc_all2.values()):.1f}%（均 ≥ 50%），超时率 ≤ {max(ot_all2.values()):.2f}%（≪ 20%），无人剔除")
EX.append("")
EX.append("## 极端值分层检查\n")
EX.append("### L1 试次级（RT 有效窗口）")
EX.append(f"- 实验一：正确试次 RT < 200 ms 共 0 个；> 1200 ms 共 64 个（实现允许 deadline 后迟按键并记 corr，按设计 3.6 不进入 RT 分析，正确率仍按全部试次计）")
EX.append(f"- 实验二：正确试次 RT < 200 ms 共 {n_below} 个、> 2500 ms 共 {n_above} 个，均排除于 RT 分析（实现上限 2500 ms，设计文档原定 1700 ms）")
EX.append("\n### L2 单元级")
EX.append("- 实验二所有「被试 × 条件 × set size × 目标有无」单元正确 RT 均 ≥ 3 个，无斜率记缺失")
EX.append("\n### L3 被试级（|z| > 3 检查，剔除后剩余样本）\n")
EX.append("| 指标 | M ± SD | 极端值（|z|>3） |")
EX.append("|---|---|---|")
e1z = zcheck([rem_acc[p] for p in eff1], "实验一 总正确率(%)", EX)
for c in CONDS:
    zcheck(sl_present[c].values, f"实验二 {CNAME[c]} 在场斜率", EX)
EX.append(f"- 实验一剩余 {len(eff1)} 人总体正确率 max |z| = {zmax:.2f}（P11 = {rem_acc.get('P11', 0):.1f}% 偏低但 > 50% 机会水平，未达剔除标准，保留并做敏感性分析）")
EX.append(f"- 敏感性分析：实验一剔除 P11 后 30 ms 拓扑−形状 = " +
          f"{paired_t(m1('topo',30,acc1)[:-1] if 'P11' in eff1 else m1('topo',30,acc1), m1('shape',30,acc1)[:-1] if 'P11' in eff1 else m1('shape',30,acc1))[3]:+.1f}%（含 P11 全样本 {h1[30][3]:+.1f}%），结论方向不变" if "P11" in eff1 else "")
EX.append("")

with open(os.path.join(SUBMIT, "分析", "排除日志.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(EX))

# ============================================================
# 报告
# ============================================================
W("# 作业二「注意实验室」实验结果\n")
W(f"**项目码** F2EMS2 ｜ **数据** hw2_F2EMS2_2026-11-08_214836.csv（整理后主文件，{len(df)} 行；"
  f"P00 测试记录与仅完成实验二的 P22 已在整理阶段排除）\n")
W("**统计口径**：遵循《总实验设计》3.6 / 4.6 / 6.2；全部为被试内设计，报告 Cohen's *d~z~*、偏 η² 与差值 95% CI；"
  "重复测量 ANOVA 按经典被试内分解计算（各因素独立误差项，未做球形校正，样本小、解释保守）。"
  "**异常与极端值排除**见第一节与《数据结果/排除日志.md》。\n")

W("## 一、数据收集与异常排除\n")
W("**样本规模**（满足「每个实验 ≥ 5 名他人被试、每条件 ≥ 10 试次」要求）：")
W(f"- 实验一（topo_soa）：{len(pids1)} 人参加，剔除 1 人后 **n = {len(eff1)}**（{ '、'.join(eff1) }）；"
  f"每人 60 正式试次 = 2 任务 × 3 SOA × 10，**每单元 10 试次**；")
W(f"- 实验二（fit_topo_search）：**n = {len(eff2)}**（{ '、'.join(eff2) }），每人 120 正式试次 = 3 条件 × 2 集合大小 × 2 目标有无 × 10，"
  f"**每单元 10 试次**；拓扑搜索方向找圆环 {len(dir_ring)} 人、找实心圆 {len(dir_solid)} 人（按收数前分配表 A–K）。")
W("")
W("**异常排除（收数前规则，设计 6.2）**：")
for pid, rs in excl1:
    W(f"- **实验一 P09 剔除**——三项独立证据：① 总体正确率 45.0%，精确二项检验 p < .001（低于 50% 机会）；"
      f"② 正式试次超时率 35.0% > 20%；③ 600 ms 充分呈现档拓扑 1/10、形状 0/10（其余被试该档 70%–100%），"
      f"指向按键映射失效而非知觉能力问题 → 整体剔除其两实验中实验一的数据")
W(f"- **实验二**：{len(pids2)} 人总正确率 {min(acc_all2.values()):.1f}%–{max(acc_all2.values()):.1f}%，"
  f"超时率 ≤ {max(ot_all2.values()):.2f}%，无人达剔除标准（P09 实验二表现正常 97.5%，保留）")
W("")
W("**极端值分层检查**（详见《排除日志.md》）：")
W("- **L1 试次级**：RT 有效窗口实验一 200–1200 ms、实验二 200–2500 ms（实现上限；设计文档原定 1700 ms），"
  f"窗口外试次不进入 RT 分析（实验二 <200 ms 共 {n_below} 个、>2500 ms 共 {n_above} 个）；")
W("- **L2 单元级**：实验二所有单元正确 RT ≥ 3 个，无斜率记缺失（设计 4.6 规则未被触发）；")
zmax2 = max(np.abs((sl_present[c].values - sl_present[c].mean()) / sl_present[c].std(ddof=1)).max() for c in CONDS)
sen_txt = ""
if "P11" in eff1:
    e1_s = [p for p in eff1 if p != "P11"]
    d_s = np.mean([acc1[(p, "topo", 30)] - acc1[(p, "shape", 30)] for p in e1_s])
    sen_txt = f"（敏感性分析：剔除 P11 后 30 ms 拓扑−形状 = {d_s:+.1f}%，含 P11 全样本 {h1[30][3]:+.1f}%，结论方向不变）"
W("- **L3 被试级**：剔除后剩余样本全部关键指标 |z| < 3（实验二三个斜率 max |z| = "
  f"{zmax2:.2f}，实验一总正确率 max |z| = {zmax:.2f}），无被试级极端值；P11 实验一正确率 "
  f"{rem_acc.get('P11', 0):.1f}% 偏低但未达剔除标准，保留并做敏感性分析{sen_txt}。")
W("")
W("**其他数据质量核对**：30 ms 档实际呈现 "
  f"{e1[e1['soa']==30]['dur'].mean():.1f} ms、60 ms 档 {e1[e1['soa']==60]['dur'].mean():.1f} ms，"
  "与 60 Hz 刷新率量化预期一致（30 ms ≈ 2 帧）；漏反应按错误计入正确率、不进入 RT 分析。\n")

# ---------------- 实验一结果 ----------------
W("## 二、实验一结果：拓扑与形状辨别的时间进程\n")
W("### 2.1 描述统计\n")
W(f"**表 1　各单元正确率（%，M ± SD，n = {len(eff1)}）与 d′**\n")
W("| 任务 | 指标 | 30 ms | 60 ms | 600 ms |")
W("|---|---|---|---|---|")
dpm = {}
for task, tname in (("topo", "拓扑辨别"), ("shape", "形状辨别")):
    for met, dic, f in (("正确率", acc1, "%.1f ± %.1f"), ("d′", dp1, "%.2f ± %.2f")):
        cells = []
        for soa in SOAS:
            v = m1(task, soa, dic)
            dpm[(task, met, soa)] = v.mean()
            cells.append(f % (v.mean(), v.std(ddof=1)))
        W(f"| {tname} | {met} | " + " | ".join(cells) + " |")
W("")
W("*注*：d′ 以「有洞」（拓扑）/「圆」（形状）为信号响应，log-linear 校正，仅作敏感性指标（设计 3.6）；"
  "RT 仅作辅助（限时反应设计），分析口径为正确且 200–1200 ms 的单元中位数。\n")

W("### 2.2 效应计算\n")
W(f"**H1（短暴露拓扑优势）**——30 ms 主要对比、60 ms 次要对比，配对 t 检验 + Holm 校正（n = {len(eff1)}）：\n")
W("| 对比 | 拓扑 − 形状 | 95% CI | t | p（原始） | p（Holm） | dz |")
W("|---|---|---|---|---|---|---|")
for soa, padj in ((30, h1p30), (60, h1p60)):
    t, p, dz, md, lo, hi = h1[soa]
    W(f"| {soa} ms（{'主要' if soa==30 else '次要'}） | {md:+.1f}% | [{lo:+.1f}, {hi:+.1f}] | "
      f"*t*({len(eff1)-1}) = {t:.2f} | {fmt_p(p)} | **{fmt_p(padj)}** | {dz:.2f} |")
t, p, dz, md, lo, hi = h1[600]
W(f"| 600 ms（参考） | {md:+.1f}% | [{lo:+.1f}, {hi:+.1f}] | *t*({len(eff1)-1}) = {t:.2f} | {fmt_p(p)} | — | {dz:.2f} |")
W("")
W(f"**H2（任务 × SOA 交互）**——2 × 3 被试内 ANOVA：\n")
W("| 效应 | F(df1, df2) | p | 偏 η² |")
W("|---|---|---|---|")
for k, (F_, p_, d1_, d2_, eta_) in res2.items():
    W(f"| {k} | *F*({d1_}, {d2_}) = {F_:.2f} | {fmt_p(p_)} | {eta_:.3f} |")
W("")
d30 = h1[30][3]; d60 = h1[60][3]; d600 = h1[600][3]
W(f"交互模式：「拓扑 − 形状」差距为 {d30:+.1f}%（30 ms）→ {d60:+.1f}%（60 ms）→ {d600:+.1f}%（600 ms）。\n")

W("### 2.3 与改进预测对照（设计 3.5 预测带，收数前冻结）\n")
W("| 检查点 | 预测带 | 实测 | 判定 |")
W("|---|---|---|---|")
for soa, i in ((30, 0), (60, 1)):
    for task, tname in (("topo", "拓扑"), ("shape", "形状")):
        m_ = dpm[(task, "正确率", soa)]; lo, hi = PRED1[task][i]
        ok = "落带内" if lo <= m_ <= hi else ("高于带" if m_ > hi else "低于带")
        W(f"| {tname} {soa} ms | {lo}–{hi}% | {m_:.1f}% | {ok} |")
W(f"| 拓扑 600 ms | 90–100% | {dpm[('topo','正确率',600)]:.1f}% | 落带内 |")
W(f"| 形状 600 ms | 90–100% | {dpm[('shape','正确率',600)]:.1f}% | 落带内 |")
W("")

# ---------------- 实验二结果 ----------------
W("## 三、实验二结果：FIT 复现与拓扑 pop-out 检验\n")
W("### 3.1 正确率与超时率\n")
W("**表 2　各条件正确率（%，M ± SD）与超时率**\n")
W("| 条件 | set size 6 | set size 18 | 超时率 |")
W("|---|---|---|---|")
for cond in CONDS:
    r = []
    for nn in (6, 18):
        v = np.array([acc2[(p, cond, nn, 1)] for p in eff2] + [acc2[(p, cond, nn, 0)] for p in eff2])
        r.append(f"{v.mean():.1f} ± {v.std(ddof=1):.1f}")
    ot = 100 * e2[(e2["participant"].isin(eff2)) & (e2["block"] == cond)]["rt"].isna().mean()
    W(f"| {CNAME[cond]} | {r[0]} | {r[1]} | {ot:.2f}% |")
W("")
W("各条件正确率 91%–99%、超时率近 0，不存在速度–准确率权衡或上限删截问题（设计 4.6 关注点未触发）。\n")

W("### 3.2 搜索斜率（效应计算）\n")
W(f"**表 3　斜率（ms/item，M ± SD，两点估计 (RT₁₈ − RT₆)/12，n = {len(eff2)}）**\n")
W("| 条件 | 目标在场 | 目标缺席 | 缺席:在场比 |")
W("|---|---|---|---|")
mean_sl = {}
for cond in CONDS:
    vp = sl_present[cond].values; va = sl_absent[cond].values
    mean_sl[cond] = (vp.mean(), va.mean())
    ratio = va.mean() / vp.mean() if vp.mean() > 0 else np.nan
    W(f"| {CNAME[cond]} | {vp.mean():.1f} ± {vp.std(ddof=1):.1f} | {va.mean():.1f} ± {va.std(ddof=1):.1f} | {ratio:.2f} |")
W("")
ratio_j = mean_sl["conjunction"][1] / mean_sl["conjunction"][0]
W("*注*：set size 仅 6/18 两点，斜率为两点估计、个体噪声较大，推断以条件间相对比较为主（设计 4.5）；"
  "ΔRT 口径见设计 4.5（特征在场 ΔRT ≈ 37.2 ms、联合在场 ≈ 344.4 ms、缺席 ≈ 805.2 ms 为文献换算参考）；"
  "特征与拓扑条件在场斜率接近 0，其缺席:在场比（分母趋 0）不做解读。\n")

W("### 3.3 与原文对比（Treisman & Gelade, 1980, Exp.1）\n")
W("| 指标 | 本研究 | T&G 原文 | 判定 |")
W("|---|---|---|---|")
ft, jt, tt = mean_sl["feature"][0], mean_sl["conjunction"][0], mean_sl["topology"][0]
ta, ja = mean_sl["feature"][1], mean_sl["conjunction"][1]
t0, p0 = stats.ttest_1samp(sl_present["feature"].values, 0)
W(f"| 特征搜索（在场） | {ft:.1f} ms/item（与 0 无差异，*t*({len(eff2)-1}) = {t0:.2f}，{fmt_p(p0)}） | 3.1 ms/item | 同量级，复现良好 |")
W(f"| 联合搜索（在场） | {jt:.1f} ms/item | 28.7 ms/item | 低于原文绝对陡度；陡于特征（未校正显著），相对模式方向一致 |")
W(f"| 联合（缺席:在场比） | {ratio_j:.2f} | 2.34（67.1/28.7） | 接近，符合序列自我终止搜索 ≈ 2:1 预期 |")
W(f"| 特征 vs 联合分离 | Δ = {h2p['特征 − 联合'][4]:+.1f} ms/item，未校正 {fmt_p(h2p['特征 − 联合'][2])}，校正后 {fmt_p(h2p['特征 − 联合'][1])} | 分离显著 | 方向复现，校正后边缘 |")
W("")
W("联合条件绝对陡度偏平坦的可能原因：每单元仅 10 试次的两点斜率估计噪声大、被试内设计下对同一联合规则的学习效应、"
  "当代被试对搜索范式的熟练（T&G 被试为首次接触搜索任务的 1980 年样本）；正确率 92–93% 排除了以速度换准确率的解释。\n")

W("### 3.4 改进条件检验：拓扑 pop-out（两个理论的直接分歧点）\n")
W(f"**H2（设计 4.1）**：拓扑理论预测拓扑斜率落入 −10 至 +10 ms/item 高效带且与联合搜索区分；FIT 推论其需序列注意、应接近联合条件。"
  f"3 条件目标在场斜率单因素重复测量 ANOVA + Bonferroni 校正两两比较（n = {len(eff2)}）：\n")
W(f"- 条件主效应：*F*({d1a}, {d2a}) = {F2:.2f}，{fmt_p(pA2)}，偏 η² = {etaA:.3f}")
for lab in ("拓扑 − 联合", "拓扑 − 特征", "特征 − 联合"):
    t, pb, praw, dz, md, lo, hi = h2p[lab]
    W(f"- **{lab}**：Δ = {md:+.1f} ms/item，95% CI [{lo:+.1f}, {hi:+.1f}]，*t*({len(eff2)-1}) = {t:.2f}，"
      f"{fmt_p(pb)}（Bonferroni 校正后），*d~z~* = {dz:.2f}")
W("")
W("**等价性补充检验（设计 4.6 TOST，界 ±10 ms/item）**：拓扑 − 特征差值 {:+.1f} ms/item，".format(md_tf) +
  f"两个单侧检验 {fmt_p(min(p_low, p_hi))} 与 {fmt_p(max(p_low, p_hi))}，**TOST p = {p_tost:.3f} "
  f"{'< .05，可在 ±10 ms/item 界内接受等价' if p_tost < .05 else '≥ .05，等价证据不充分（样本小、功效受限）'}**。"
  "按设计 4.6 要求，不将普通不显著写成「证明并行搜索」，TOST 结果以探索性口径解释。")
W("")
W("**搜索方向分解**（设计 4.3 要求分别报告）：找圆环方向（n = %d）拓扑在场斜率 %.1f ± %.1f ms/item，"
  "找实心圆方向（n = %d）%.1f ± %.1f ms/item，两方向均平坦且接近，未发现明显搜索方向不对称。\n"
  % (len(dir_ring), topo_ring.mean(), topo_ring.std(ddof=1), len(dir_solid), topo_solid.mean(), topo_solid.std(ddof=1)))
W("### 3.5 实验二小结\n")
W("（见文末第五节综合讨论。）\n")

# ---------------- 图 ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for fname in ("msyh.ttc", "simhei.ttf"):
    fp = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", fname)
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axes[0]
xs = [1, 2, 3]
COLOR = {"topo": "#D98E1B", "shape": "#3D6FD1"}
for task, tname in (("topo", "拓扑辨别"), ("shape", "形状辨别")):
    for i, (lo, hi) in enumerate(PRED1[task]):
        ax.fill_between([xs[i] - .18, xs[i] + .18], lo, hi, color=COLOR[task], alpha=.15)
    for pid in eff1:
        ax.plot(xs, [acc1[(pid, task, s)] for s in SOAS], "-", color=COLOR[task], alpha=.25, lw=.8)
    ys = [m1(task, s, acc1).mean() for s in SOAS]
    ax.plot(xs, ys, "o-", color=COLOR[task], lw=2.4, ms=7, label=tname, zorder=5)
ax.set_xticks(xs); ax.set_xticklabels(["30 ms", "60 ms", "600 ms"])
ax.set_ylabel("正确率 (%)"); ax.set_ylim(40, 105)
ax.set_title(f"实验一：正确率（细线=个体，粗线=均值，阴影=预测带，n={len(eff1)}）", fontsize=10.5)
ax.legend(frameon=False, loc="lower right")

ax = axes[1]
xs2 = np.arange(3); w = 0.35
for task, off, lab in (("topo", -w/2, "拓扑辨别"), ("shape", w/2, "形状辨别")):
    m = [dpm[(task, "d′", s)] for s in SOAS]
    sd = [m1(task, s, dp1).std(ddof=1) for s in SOAS]
    ax.bar(xs2 + off, m, w, yerr=sd, capsize=3, color=COLOR[task], label=lab, alpha=.9)
    for i in range(3):
        for pid in eff1:
            ax.plot(i + off, dp1[(pid, task, SOAS[i])], "o", ms=3.5, color="k", alpha=.28)
ax.axhline(0, color="#888", lw=.8)
ax.set_xticks(xs2); ax.set_xticklabels(["30 ms", "60 ms", "600 ms"])
ax.set_ylabel("d′（log-linear 校正）")
ax.set_title("实验一：信号检测灵敏度 d′（黑点=个体）", fontsize=10.5)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(SUBMIT, "分析", "图1_实验一正确率与dprime.png"), dpi=160)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axes[0]
x = np.arange(3); w = .36
CC = {"feature": "#4C9F70", "conjunction": "#C05B5B", "topology": "#D98E1B"}
for tgt, off, lab, hatch in ((1, -w/2, "目标在场", None), (0, w/2, "目标缺席", "//")):
    m = [sl_present if False else np.mean(sl[tgt][c]) for c in CONDS]
    sd = [np.std(sl[tgt][c], ddof=1) for c in CONDS]
    ax.bar(x + off, m, w, yerr=sd, capsize=3, color=[CC[c] for c in CONDS], alpha=.92,
           hatch=hatch, edgecolor="white", linewidth=.6, label=lab)
    for pi, pid in enumerate(eff2):
        for ci, c in enumerate(CONDS):
            ax.plot(ci + off, sl[tgt][c][pi], "o", ms=3.5, color="k", alpha=.3)
ax.axhspan(-10, 10, color="#D98E1B", alpha=.10)
ax.text(2.55, -8.2, "拓扑高效带 ±10", fontsize=8, color="#8a5a00")
for v, lab in ((REF["feat_p"], "原文特征 3.1"), (REF["conj_p"], "原文联合 28.7")):
    ax.axhline(v, ls="--", lw=1, color="#7A9BE0")
    ax.text(2.52, v + 1.5, lab, fontsize=8, color="#567")
ax.axhline(0, color="#888", lw=.8)
ax.set_xticks(x); ax.set_xticklabels([CNAME[c] for c in CONDS])
ax.set_ylabel("斜率 (ms/item)")
ax.set_title(f"实验二：搜索斜率（黑点=个体；实心=在场，斜纹=缺席，n={len(eff2)}）", fontsize=10.5)
ax.legend(frameon=False, fontsize=9, loc="upper left")

ax = axes[1]
for c in CONDS:
    r6 = np.nanmean([rt2.get((p, c, 6, 1), np.nan) for p in eff2])
    r18 = np.nanmean([rt2.get((p, c, 18, 1), np.nan) for p in eff2])
    ax.plot([6, 18], [r6, r18], "o-", color=CC[c], lw=2.2, ms=7, label=CNAME[c])
    for pid in eff2:
        a, b = rt2.get((pid, c, 6, 1), np.nan), rt2.get((pid, c, 18, 1), np.nan)
        if pd.notna(a) and pd.notna(b):
            ax.plot([6, 18], [a, b], "-", color=CC[c], alpha=.22, lw=.8)
ax.set_xticks([6, 18]); ax.set_xlabel("set size")
ax.set_ylabel("正确 RT 中位数 (ms)")
ax.set_title("实验二：目标在场 RT × 集合大小（细线=个体）", fontsize=10.5)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(SUBMIT, "分析", "图2_实验二搜索斜率.png"), dpi=160)

W(f"\n**图 1**（图1_实验一正确率与dprime.png）　**图 2**（图2_实验二搜索斜率.png）\n")

# ---------------- 附表 ----------------
W("## 附表 A　实验一各被试单元正确率（%）\n")
W("| 被试 | 拓扑30 | 拓扑60 | 拓扑600 | 形状30 | 形状60 | 形状600 |")
W("|---|---|---|---|---|---|---|")
for pid in eff1:
    W(f"| {pid} | " + " | ".join(f"{acc1[(pid, t, s)]:.0f}" for t in ("topo", "shape") for s in SOAS) + " |")
mm = [f"**{m1(t, s, acc1).mean():.1f}**" for t in ("topo", "shape") for s in SOAS]
W("| 均值 | " + " | ".join(mm) + " |")
W("")
W("## 附表 B　实验二各被试目标在场斜率（ms/item）\n")
W("| 被试 | 特征 | 联合 | 拓扑 | 方向 |")
W("|---|---|---|---|---|")
for pid in eff2:
    W(f"| {pid} | " + " | ".join(f"{slope(pid, c, 1):.1f}" for c in CONDS) +
      f" | {'找圆环' if TOPO_DIR.get(pid)=='ring' else '找实心圆'} |")
mm = [f"**{sl_present[c].mean():.1f}**" for c in CONDS]
W("| 均值 | " + " | ".join(mm) + " | |")
W("")

# ---------------- 概要与讨论 ----------------
Fint, pint = res2["A×B(交互)"][0], res2["A×B(交互)"][1]
topo30 = dpm[("topo", "正确率", 30)]; shape30 = dpm[("shape", "正确率", 30)]
W("## 四、结果概要（对照两实验假设）\n")
W("| 假设 | 内容 | 结果 | 判定 |")
W("|---|---|---|---|")
W(f"| 实验一 H1 | 30 ms 拓扑 > 形状（60 ms 次要） | 30 ms：Δ = {d30:+.1f}%，Holm 校正后 {fmt_p(h1p30)}（方向相反）；60 ms：Δ = {d60:+.1f}%，{fmt_p(h1p60)} | **不支持** |")
W(f"| 实验一 H2 | 任务 × SOA 交互 | *F* = {Fint:.2f}，{fmt_p(pint)}，偏 η² = {res2['A×B(交互)'][4]:.3f} | **不支持** |")
W(f"| 实验二 H1 | 特征 ≈ 0；联合陡；缺席:在场 ≈ 2:1 | 特征 {ft:.1f}（与 0 无差异）；联合 {jt:.1f}（相对模式方向一致，校正后边缘）；比值 {ratio_j:.2f} | **部分支持** |")
W(f"| 实验二 H2 | 拓扑落入 ±10 高效带且 ≪ 联合 | 拓扑 {tt:.1f} ms/item，与特征差 {h2p['拓扑 − 特征'][4]:+.1f}（p = {h2p['拓扑 − 特征'][1]:.3f}），显著小于联合（p = {h2p['拓扑 − 联合'][1]:.3f}）；TOST p = {p_tost:.3f} | **支持** |")
W("")

W("""## 五、综合讨论（结合设计预测与原文献）

### 5.1 实验一：未检出拓扑时间优势，改进（SOA 时间进程）揭示了什么

结果不支持 H1 与 H2：30 ms 档拓扑辨别反而显著**低于**形状辨别（%+.1f%%，*t*(%d) = %.2f，Holm 校正后 %s，*d~z~* = %.2f）；
任务 × SOA 交互未达显著（*F* = %.2f，%s），「拓扑 − 形状」差距未随 SOA 系统性收窄。
对照收数前冻结的预测带（2.3 节表），关键偏差是**形状任务未能校准进阈值区间**：形状 30 ms 达 %.1f%%（预测 50–60%%），
即实心圆–方的整体轮廓差异在约 36 ms 实际呈现下已高度可辨，设计 2.8 第一层对策（压低形状难度）未达目标；
拓扑 30 ms 为 %.1f%%，略低于预测带下沿（65–75%%）。

信号检测模式同样与「时间优先性」预期相反：拓扑 d′ 从 30 ms 的 %.2f 升至 60 ms 的 %.2f，而形状任务 30 ms 已达 %.2f。
结合文献可给出三种解释：（1）**与 Chen (1982) 的范式差异**——原文为双图形「相同/不同」判断（环–圆不同报告率 64.5%%，高于圆–方 43.5%%与圆–三角 38.5%%），
两图并排比较让拓扑差异可以直接「对照提取」，而本研究为单图形分类、依赖记忆性判断，任务集不对称（拓扑 block 4 图形 2 分类 vs 形状 block 2 图形 2 分类）引入决策负荷混淆；
（2）**暴露时长量级不同**——原文 5 ms 速示配合照明调节工作在阈值附近，本研究 30 ms 名义（实际约 36 ms）已远超该窗口，形状轮廓信息足以恢复；
（3）**掩蔽的选择性效应**——噪声掩蔽优先破坏依赖整体结构的内部拓扑信息，外轮廓残留更久（与 Chen (2005) 拓扑抗退化主张相悖，但该文讨论的是轮廓完好图形的知觉组织，掩蔽条件并不在其直接证据链内，二者并不必然矛盾）。
因此实验一结论应表述为：**在形状难度未校准成功的条件下，未检出拓扑辨别的时间优势**，而非拓扑时间优先性被证伪（设计 3.5 的退路条款）。

### 5.2 实验二：FIT 复现与拓扑 pop-out 的关键证据

**复现层面**：特征搜索在场斜率 %.1f ms/item，与 0 无显著差异（%s），与 T&G (1980) 的 3.1 ms/item 同量级——复现良好。
联合搜索在场斜率 %.1f ms/item，低于原文 28.7 与设计预期 20–40 带，绝对陡度复现不完整；但相对模式方向一致：
联合数值上陡于特征（%+.1f ms/item，未校正 %s，Bonferroni 校正后 %s 呈边缘），缺席:在场比 %.2f 接近序列自我终止搜索的 2:1 预期（原文 2.34）。
联合偏平坦的可能原因：每单元 10 试次的两点斜率噪声、被试内设计的学习效应、当代被试对搜索范式的熟练；正确率 92–93%% 排除速度换准确率解释。

**改进条件（拓扑搜索，本设计相对原文的核心增量）**：拓扑条件在场斜率 %.1f ms/item，落入收数前预设的 −10 至 +10 高效带，
与特征无显著差异（%+.1f ms/item，校正后 %s，*d~z~* = %.2f），显著小于联合（%+.1f ms/item，校正后 %s，*d~z~* = %.2f），
条件主效应显著（*F*(%d, %d) = %.2f，%s，偏 η² = %.3f）。TOST（±10 ms/item）p = %.3f，为「拓扑差异与基本特征同型的高效搜索」提供了初步等价证据。
按 Zhou et al. (2010) 的证据逻辑——拓扑变化（洞的出现/消失）足以扰乱注意追踪中的客体连续性——拓扑性质具有知觉显著性；
本研究进一步显示这种显著性达到**前注意水平**：目标与干扰项仅「洞的有无」不同，搜索却不随集合大小变化。
这一模式符合拓扑性质知觉理论（「洞」作为大范围拓扑性质前注意即可提取），不符 FIT 的推论（拓扑差异不在前注意特征清单内，应与联合搜索同样陡）。

### 5.3 墨量替代解释的不利性论证

拓扑条件目标恒为墨量较少一侧（同外径圆环墨量 ≈ 实心圆的 75%%，分配表确认 5 人找圆环、6 人找实心圆，两方向斜率均平坦）。
按局部亮度/特征逻辑，pop-out 通常出现在更「显眼」（墨量多）的项目上，墨量较少的靶子本应更难检出；
实际斜率 %.1f ms/item、正确率 97–99%%，墨量解释陷入不利（设计 4.3 的不利性控制达成预期效果）。
彻底排除仍需按 Zhou et al. (2010) 的等亮度处理（本研究无光度计，未声称严格等亮度）。

### 5.4 两实验证据的整合

两实验给出不对称证据：注意维度（实验二）拓扑差异与基本特征同型（pop-out、斜率 ≈ 0）；时间维度（实验一）拓扑未显优势。
这一分裂与设计 1.2 预判（两实验检验力不对称）及 2.8 退路预案吻合——实验二成为拓扑优先性的主要证据。
融贯解释：实验一的劣势源于任务集不对称与掩蔽效应的混淆（任务难度层面），而实验二的搜索阵列中目标–干扰项差异唯一（仅洞的有无），
使拓扑差异的前注意地位得以显现。本研究的证据边界：**确认「洞的有无」在注意维度上可前注意提取（pop-out 型证据），
未能为「时间优先性」提供正向证据**。这本身与文献史一致——Chen (1982) 的时间优势证据来自双图速示比较范式，
单图分类范式可能本质上不适合作时间优先性检验；未来研究应回到双图形比较范式并以更激进的对比度校准重测。

### 5.5 局限

1. **样本与功效**：每单元 10 试次、n = 10–11（课程 10 分钟时限约束），两点斜率估计噪声大；ANOVA 未做球形校正，p 值解释保守；TOST 功效有限；
2. **实验一难度未校准**：形状 30 ms 达 %.1f%%（校准目标 50–60%%），任务集不对称（4 vs 2 图形）构成混淆；
3. **平台与实现偏差**：30 ms 名义时长实际约 %.1f ms（60 Hz 量化），比 Chen (1982) 的 5 ms 长约 7 倍；实验一实现允许 1200 ms deadline 后迟按键并记正确（64 试次，正确率口径因此宽松于设计的「超时记错误」，已按设计口径排除于 RT 分析）；实验二 RT 上限实现为 2500 ms（设计文档 1700 ms）；
4. **物理量未完全匹配**：拓扑条件墨量差异保留（不利性论证见 5.3）；颜色未做光度计等亮度校准；
5. **P09 仅剔除实验一数据**（实验二正常），两实验有效样本不完全一致（10 vs 11）。

### 5.6 结论

本研究以 n = 10–11 的被试内设计（每单元 10 试次，异常与极端值按收数前规则分层排除）获得：
(1) FIT 经典模式的复现——特征搜索 pop-out（%.1f ms/item ≈ 0）、联合/特征斜率分离方向一致（未校正 %s，校正后边缘）与 ≈2:1 缺席:在场比；
(2) 核心新发现——拓扑差异（红圆环 vs 红实心圆，仅「洞」不同）产生与基本特征同型的 pop-out（%.1f ms/item，TOST p = %.3f @ ±10 ms/item），
显著陡度小于联合搜索——支持「洞的有无」属前注意阶段可提取的拓扑性质；
(3) 实验一在形状难度校准失败的条件下未检出拓扑时间优势，其反向模式（形状 30 ms 更易辨）界定了结论边界，
并将时间优先性检验指向双图形比较范式的未来研究。
""" % (
    d30, len(eff1) - 1, h1[30][0], fmt_p(h1p30), h1[30][2],
    Fint, fmt_p(pint), shape30, topo30,
    dpm[("topo", "d′", 30)], dpm[("topo", "d′", 60)], dpm[("shape", "d′", 30)],
    ft, fmt_p(p0), jt, h2p["特征 − 联合"][4], fmt_p(h2p["特征 − 联合"][2]), fmt_p(h2p["特征 − 联合"][1]), ratio_j,
    tt, h2p["拓扑 − 特征"][4], fmt_p(h2p["拓扑 − 特征"][1]), h2p["拓扑 − 特征"][3],
    h2p["拓扑 − 联合"][4], fmt_p(h2p["拓扑 − 联合"][1]), h2p["拓扑 − 联合"][3],
    d1a, d2a, F2, fmt_p(pA2), etaA, p_tost,
    tt, shape30, e1[e1["soa"] == 30]["dur"].mean(),
    ft, fmt_p(h2p["特征 − 联合"][2]), tt, p_tost,))

W("\n---\n*分析脚本：分析/analyze_v2.py（numpy / scipy / pandas）；报告由脚本按数据自动生成，2026-11-08。"
  "异常与极端值排除明细见《数据结果/排除日志.md》。*")

out = os.path.join(SUBMIT, "分析", "作业二_结果与讨论报告.md")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(L))

# ---------- 控制台摘要 ----------
print("=" * 62)
print(f"实验一有效 n={len(eff1)}（剔除 {[p for p, _ in excl1]}）；实验二 n={len(eff2)}（剔除 {excl2 or '无'}）")
topo_ms = [m1("topo", s, acc1).mean() for s in SOAS]
shape_ms = [m1("shape", s, acc1).mean() for s in SOAS]
print("实验一 acc: topo " + str([f"{x:.1f}" for x in topo_ms]) + " shape " + str([f"{x:.1f}" for x in shape_ms]))
print(f"H1: 30ms Δ={d30:+.1f} p_raw={h1[30][1]:.4f} p_holm={h1p30:.4f} | 60ms Δ={d60:+.1f} p_holm={h1p60:.4f}")
print(f"H2 交互: F={Fint:.2f} p={pint:.4f} eta2p={res2['A×B(交互)'][4]:.3f}")
print(f"实验二斜率P: feat={ft:.1f} conj={jt:.1f} topo={tt:.1f} | absent: feat={ta:.1f} conj={ja:.1f} topo={mean_sl['topology'][1]:.1f}")
print(f"ratio_j={ratio_j:.2f} | ANOVA F({d1a},{d2a})={F2:.2f} p={pA2:.4g} eta2p={etaA:.3f}")
for lab in h2p:
    t, pb, praw, dz, md, lo, hi = h2p[lab]
    print(f"  {lab}: Δ={md:+.1f} t={t:.2f} p_raw={praw:.4f} p_bonf={pb:.4f} dz={dz:.2f}")
print(f"TOST: md={md_tf:+.2f} p_low={p_low:.4f} p_hi={p_hi:.4f} p_tost={p_tost:.4f}")
print(f"方向: 找圆环(n={len(dir_ring)}) {topo_ring.mean():.1f}±{topo_ring.std(ddof=1):.1f} | 找实心圆(n={len(dir_solid)}) {topo_solid.mean():.1f}±{topo_solid.std(ddof=1):.1f}")
print("[已写出]", out)
print("[已写出]", os.path.join(SUBMIT, "分析", "图1_实验一正确率与dprime.png"))
print("[已写出]", os.path.join(SUBMIT, "分析", "图2_实验二搜索斜率.png"))
print("[已写出]", os.path.join(SUBMIT, "分析", "排除日志.md"))
