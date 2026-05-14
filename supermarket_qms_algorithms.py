"""
Supermarket Queue Management System — OS Scheduling Algorithms
==============================================================
Simulates supermarket checkout queues as OS scheduling problems.

Mapping:
  Customer  → Process
  Basket    → Burst Time
  Arrival   → Arrival Time
  Counter   → CPU
  Switch    → Context Switch
  Express   → High Priority

Algorithms implemented:
  1. FCFS        — First Come First Served
  2. SJN         — Shortest Job Next (non-preemptive)
  3. SRTF        — Shortest Remaining Time First (preemptive)
  4. Round Robin — Time-sliced with configurable quantum
  5. Multilevel  — Express lane (RR) + Standard lane (FCFS)
"""

import math
import copy

# Context switch penalty (time units)
CONTEXT_SWITCH_TIME = 0.5


# ──────────────────────────────────────────────────────────────
# SEEDED RANDOM NUMBER GENERATOR
# ──────────────────────────────────────────────────────────────

def seeded_rand(seed):
    """Linear Congruential Generator — matches JS implementation."""
    state = [seed & 0xFFFFFFFF]

    def rand():
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return state[0] / 4294967296

    return rand


def exp_rand(r, rate):
    """Exponential random variate."""
    return -math.log(1 - r()) / rate


# ──────────────────────────────────────────────────────────────
# CUSTOMER GENERATION
# ──────────────────────────────────────────────────────────────

def gen_customers(n, seed=42):
    """
    Generate n customers with exponential inter-arrival times.

    Each customer dict contains:
      id       : int   — 1-indexed customer ID
      arrival  : float — arrival time
      items    : int   — number of items (burst time)
      priority : int   — 1=regular, 2=senior, 3=staff
      lane     : str   — 'express' (<5 items) or 'standard'

    Parameters
    ----------
    n    : number of customers
    seed : RNG seed (default 42)
    """
    r = seeded_rand(seed)
    customers = []
    t = 0.0

    for i in range(n):
        t += exp_rand(r, 0.4)
        roll = r()
        if roll < 0.4:
            items = max(1, int(r() * 4) + 1)
        elif roll < 0.75:
            items = int(r() * 10) + 6
        else:
            items = int(r() * 20) + 16

        pr = 1 if r() < 0.7 else (2 if r() < 0.9 else 3)
        customers.append({
            "id": i + 1,
            "arrival": round(t, 2),
            "items": items,
            "priority": pr,
            "lane": "express" if items < 5 else "standard",
        })

    return customers


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def reset_cust(c):
    """Add simulation fields (remaining, startTime, finishTime) to a customer."""
    return {**c, "remaining": c["items"], "startTime": None, "finishTime": None}


def deep_copy(customers):
    """Deep-copy customer list and reset simulation state."""
    return [reset_cust(c) for c in customers]


def calc_metrics(res):
    """
    Compute performance metrics from a simulation result.

    Returns a dict with keys:
      avg_wait   : average waiting time
      avg_tat    : average turnaround time (including ctx penalty)
      throughput : customers served per time unit
      util       : CPU/counter utilisation (%)
      ctx        : number of context switches
    """
    s = res["served"]
    if not s:
        return {"avg_wait": 0, "avg_tat": 0, "throughput": 0, "util": 0, "ctx": 0}

    avg_wait = sum(c["startTime"] - c["arrival"] for c in s) / len(s)
    avg_tat = sum(
        (c["finishTime"] - c["arrival"] + res["ctx_penalty"] / len(s)) for c in s
    ) / len(s)
    throughput = len(s) / res["total"] if res["total"] > 0 else 0

    return {
        "avg_wait": avg_wait,
        "avg_tat": avg_tat,
        "throughput": throughput,
        "util": res["util"],
        "ctx": res["ctx_switches"],
    }


# ──────────────────────────────────────────────────────────────
# ALGORITHM 1 — FCFS (First Come First Served)
# ──────────────────────────────────────────────────────────────

def fcfs(customers, n_counters):
    """
    First Come First Served — non-preemptive.

    Customers are served in arrival order. Each idle counter picks
    the earliest-arrived customer that has already arrived.

    Parameters
    ----------
    customers  : list of customer dicts (from gen_customers)
    n_counters : number of checkout counters (= number of CPUs)

    Returns
    -------
    dict with keys: served, total, ctx_switches, ctx_penalty, util, timeline
    """
    custs = deep_copy(customers)
    custs.sort(key=lambda c: c["arrival"])

    counters = [{"id": i, "busy": False, "busy_time": 0, "serving": None}
                for i in range(n_counters)]

    in_prog = {}      # counter_id → {cust, end}
    queue   = list(custs)
    served  = []
    timeline = []
    now = 0.0

    while queue or in_prog:
        # Complete finished jobs
        for cid in list(in_prog):
            cust, end = in_prog[cid]["cust"], in_prog[cid]["end"]
            if now >= end:
                cust["finishTime"] = end
                counters[cid]["busy_time"] += cust["items"]
                counters[cid]["busy"] = False
                counters[cid]["serving"] = None
                served.append(cust)
                del in_prog[cid]

        # Assign waiting customers to free counters
        for c in counters:
            if c["busy"]:
                continue
            idx = next((i for i, cu in enumerate(queue) if cu["arrival"] <= now), -1)
            if idx == -1:
                break
            cust = queue.pop(idx)
            cust["startTime"] = now
            end_t = now + cust["items"]
            in_prog[c["id"]] = {"cust": cust, "end": end_t}
            timeline.append({"custId": cust["id"], "counterId": c["id"],
                              "start": now, "end": end_t})
            c["busy"] = True
            c["serving"] = cust["id"]

        # Advance time to next event
        events = ([v["end"] for v in in_prog.values()] +
                  [cu["arrival"] for cu in queue if cu["arrival"] > now])
        if not events:
            break
        now = min(events)

    total = max((c["finishTime"] for c in served), default=0)
    total_busy = sum(c["busy_time"] for c in counters)
    util = (total_busy / (total * n_counters) * 100) if total > 0 else 0

    return {
        "served": served,
        "total": total,
        "ctx_switches": 0,
        "ctx_penalty": 0,
        "util": util,
        "timeline": timeline,
    }


# ──────────────────────────────────────────────────────────────
# ALGORITHM 2 — SJN (Shortest Job Next / SJF non-preemptive)
# ──────────────────────────────────────────────────────────────

def sjn(customers, n_counters):
    """
    Shortest Job Next — non-preemptive (also known as SJF).

    Among all arrived customers, the one with the fewest items is
    served next. Potential starvation for large-basket customers.

    Parameters
    ----------
    customers  : list of customer dicts
    n_counters : number of checkout counters

    Returns
    -------
    dict with keys: served, total, ctx_switches, ctx_penalty, util, timeline
    """
    custs = deep_copy(customers)
    custs.sort(key=lambda c: c["arrival"])

    counters = [{"id": i, "busy": False, "busy_time": 0}
                for i in range(n_counters)]

    waiting     = []
    not_arrived = list(custs)
    in_prog     = {}
    served      = []
    timeline    = []
    now = 0.0

    while not_arrived or waiting or in_prog:
        # Admit newly arrived customers to waiting list
        batch = [c for c in not_arrived if c["arrival"] <= now]
        if batch:
            waiting.extend(batch)
            not_arrived = [c for c in not_arrived if c["arrival"] > now]
            waiting.sort(key=lambda c: c["items"])  # SJN: shortest first

        # Complete finished jobs
        for cid in list(in_prog):
            cust, end = in_prog[cid]["cust"], in_prog[cid]["end"]
            if now >= end:
                cust["finishTime"] = end
                counters[cid]["busy_time"] += cust["items"]
                counters[cid]["busy"] = False
                served.append(cust)
                del in_prog[cid]

        # Assign shortest job to free counters
        for c in counters:
            if c["busy"] or not waiting:
                continue
            cust = waiting.pop(0)
            cust["startTime"] = now
            end_t = now + cust["items"]
            in_prog[c["id"]] = {"cust": cust, "end": end_t}
            timeline.append({"custId": cust["id"], "counterId": c["id"],
                              "start": now, "end": end_t})
            c["busy"] = True

        events = ([v["end"] for v in in_prog.values()] +
                  [c["arrival"] for c in not_arrived])
        if not events:
            break
        now = min(events)

    total = max((c["finishTime"] for c in served), default=0)
    total_busy = sum(c["busy_time"] for c in counters)
    util = (total_busy / (total * n_counters) * 100) if total > 0 else 0

    return {
        "served": served,
        "total": total,
        "ctx_switches": 0,
        "ctx_penalty": 0,
        "util": util,
        "timeline": timeline,
    }


# ──────────────────────────────────────────────────────────────
# ALGORITHM 3 — SRTF (Shortest Remaining Time First, preemptive)
# ──────────────────────────────────────────────────────────────

def srtf(customers, n_counters):
    """
    Shortest Remaining Time First — preemptive SJF.

    A running customer is preempted whenever a new arrival has
    fewer remaining items. High starvation risk for large baskets.

    Parameters
    ----------
    customers  : list of customer dicts
    n_counters : number of checkout counters

    Returns
    -------
    dict with keys: served, total, ctx_switches, ctx_penalty, util, timeline
    """
    custs = deep_copy(customers)
    custs.sort(key=lambda c: c["arrival"])

    not_arrived = list(custs)
    ready_q     = []
    counters    = [{"id": i, "current": None} for i in range(n_counters)]
    served      = []
    timeline    = []
    ctx_switches = 0
    now = 0.0

    def arrive_until(t):
        batch = [c for c in not_arrived if c["arrival"] <= t]
        if batch:
            for c in batch:
                ready_q.append(c)
                not_arrived.remove(c)
            ready_q.sort(key=lambda c: c["remaining"])

    def next_evt():
        e = ([c["current"]["endSlice"] for c in counters if c["current"]] +
             [c["arrival"] for c in not_arrived if c["arrival"] > now])
        return min(e) if e else math.inf

    def fill():
        for c in counters:
            if c["current"] or not ready_q:
                continue
            cu = ready_q.pop(0)
            if cu["startTime"] is None:
                cu["startTime"] = now
            nxt = next_evt()
            h = nxt if math.isfinite(nxt) else now + cu["remaining"]
            sl = min(cu["remaining"], max(0, h - now))
            c["current"] = {"cust": cu, "startSlice": now, "endSlice": now + sl}

    def preempt():
        nonlocal ctx_switches
        if not ready_q:
            return
        for c in counters:
            if not c["current"]:
                continue
            if ready_q[0]["remaining"] < c["current"]["cust"]["remaining"]:
                cust      = c["current"]["cust"]
                start_sl  = c["current"]["startSlice"]
                elapsed   = now - start_sl
                if elapsed > 0:
                    timeline.append({"custId": cust["id"], "counterId": c["id"],
                                     "start": start_sl, "end": now})
                    cust["remaining"] -= elapsed
                    cust["remaining"] = max(0, round(cust["remaining"], 6))
                ready_q.append(cust)
                ready_q.sort(key=lambda c: c["remaining"])
                ctx_switches += 1
                c["current"] = None

    arrive_until(0)
    it = 0

    while (not_arrived or ready_q or any(c["current"] for c in counters)) and it < 200_000:
        it += 1
        preempt()
        fill()

        nxt = next_evt()
        if not math.isfinite(nxt):
            break
        now = nxt

        for c in counters:
            if not c["current"]:
                continue
            cu, start_sl, end_sl = (c["current"]["cust"],
                                    c["current"]["startSlice"],
                                    c["current"]["endSlice"])
            if now < end_sl:
                continue
            timeline.append({"custId": cu["id"], "counterId": c["id"],
                              "start": start_sl, "end": end_sl})
            cu["remaining"] -= (end_sl - start_sl)
            cu["remaining"] = max(0, round(cu["remaining"], 6))
            if cu["remaining"] <= 0:
                cu["finishTime"] = end_sl
                served.append(cu)
            else:
                ready_q.append(cu)
                ready_q.sort(key=lambda c: c["remaining"])
                ctx_switches += 1
            c["current"] = None

        arrive_until(now)

        # Recompute slice end for still-running jobs
        nn = next_evt()
        for c in counters:
            if not c["current"]:
                continue
            cu = c["current"]["cust"]
            h  = nn if math.isfinite(nn) else now + cu["remaining"]
            sl = min(cu["remaining"], max(0, h - now))
            c["current"]["startSlice"] = now
            c["current"]["endSlice"]   = now + sl

    ctx_penalty = ctx_switches * CONTEXT_SWITCH_TIME
    raw_end     = max((c["finishTime"] for c in served), default=0)
    total       = raw_end + ctx_penalty
    total_busy  = sum(c["items"] for c in served)
    util = min(100, (total_busy / (total * n_counters) * 100)) if total > 0 else 0

    return {
        "served": served,
        "total": total,
        "ctx_switches": ctx_switches,
        "ctx_penalty": ctx_penalty,
        "util": util,
        "timeline": timeline,
    }


# ──────────────────────────────────────────────────────────────
# ALGORITHM 4 — Round Robin
# ──────────────────────────────────────────────────────────────

def round_robin(customers, n_counters, quantum=5):
    """
    Round Robin — preemptive, time-sliced.

    Each customer is served for at most `quantum` time units before
    being returned to the back of the ready queue. No starvation.

    Parameters
    ----------
    customers  : list of customer dicts
    n_counters : number of checkout counters
    quantum    : time slice (default 5)

    Returns
    -------
    dict with keys: served, total, ctx_switches, ctx_penalty, util, timeline
    """
    custs = deep_copy(customers)
    custs.sort(key=lambda c: c["arrival"])

    counters = [{"id": i, "busy": False, "busy_time": 0}
                for i in range(n_counters)]

    ready_q    = []
    rem_arr    = list(custs)
    in_prog    = {}
    served     = []
    timeline   = []
    ctx_switches = 0
    now = 0.0

    def enqueue():
        arrived = [c for c in rem_arr if c["arrival"] <= now]
        for c in arrived:
            ready_q.append(c)
            rem_arr.remove(c)

    enqueue()
    it = 0

    while (rem_arr or ready_q or in_prog) and it < 200_000:
        it += 1

        # Complete finished slices
        for cid in list(in_prog):
            entry = in_prog[cid]
            cust, start, end = entry["cust"], entry["start"], entry["end"]
            if now >= end:
                done = end - start
                cust["remaining"] -= done
                cust["remaining"] = max(0, round(cust["remaining"], 6))
                counters[cid]["busy_time"] += done
                counters[cid]["busy"] = False
                timeline.append({"custId": cust["id"], "counterId": cid,
                                  "start": start, "end": end})
                del in_prog[cid]
                if cust["remaining"] <= 0:
                    cust["finishTime"] = end
                    served.append(cust)
                else:
                    ctx_switches += 1
                    ready_q.append(cust)

        enqueue()

        # Assign time slices to free counters
        for c in counters:
            if c["busy"] or not ready_q:
                continue
            cust = ready_q.pop(0)
            if cust["startTime"] is None:
                cust["startTime"] = now
            sl = min(quantum, cust["remaining"])
            in_prog[c["id"]] = {"cust": cust, "start": now, "end": now + sl}
            c["busy"] = True

        events = ([v["end"] for v in in_prog.values()] +
                  [c["arrival"] for c in rem_arr])
        if not events:
            if ready_q:
                now += 0.001
            else:
                break
        else:
            now = min(events)

    ctx_penalty = ctx_switches * CONTEXT_SWITCH_TIME
    raw_end     = max((c["finishTime"] for c in served), default=0)
    total       = raw_end + ctx_penalty
    total_busy  = sum(c["busy_time"] for c in counters)
    util = (total_busy / (total * n_counters) * 100) if total > 0 else 0

    return {
        "served": served,
        "total": total,
        "ctx_switches": ctx_switches,
        "ctx_penalty": ctx_penalty,
        "util": util,
        "timeline": timeline,
    }


# ──────────────────────────────────────────────────────────────
# ALGORITHM 5 — Multilevel Queue
# ──────────────────────────────────────────────────────────────

def multilevel(customers, n_counters, quantum=5):
    """
    Multilevel Queue — two fixed queues, no migration.

    Express lane  (<5 items) : 60% of counters, scheduled with RR
    Standard lane (≥5 items) : remaining counters, scheduled with FCFS

    Parameters
    ----------
    customers  : list of customer dicts
    n_counters : number of checkout counters
    quantum    : RR quantum for express lane (default 5)

    Returns
    -------
    dict with extra keys: lanes (express/standard counts and counter split)
    """
    express_counters  = max(1, int(n_counters * 0.6))
    standard_counters = max(1, n_counters - express_counters)

    express  = [c for c in customers if c["items"] < 5]
    standard = [c for c in customers if c["items"] >= 5]

    rr_res   = round_robin(express,  express_counters,  quantum)
    fcfs_res = fcfs(standard, standard_counters)

    # Merge timelines (shift standard counter IDs)
    merged_tl = (
        [{**t, "lane": "express"} for t in rr_res["timeline"]] +
        [{**t, "counterId": t["counterId"] + express_counters, "lane": "standard"}
         for t in fcfs_res["timeline"]]
    )

    all_served = (
        [{**c, "lane": "express"}  for c in rr_res["served"]] +
        [{**c, "lane": "standard"} for c in fcfs_res["served"]]
    )
    all_served.sort(key=lambda c: c["id"])

    rr_end   = max((c["finishTime"] for c in rr_res["served"]),   default=0)
    fcfs_end = max((c["finishTime"] for c in fcfs_res["served"]), default=0)
    total_raw = max(rr_end, fcfs_end)

    ctx_switches = rr_res["ctx_switches"] + fcfs_res["ctx_switches"]
    ctx_penalty  = ctx_switches * CONTEXT_SWITCH_TIME
    total        = total_raw + ctx_penalty

    total_busy = sum(c["items"] for c in all_served)
    util = min(100, (total_busy / (total * n_counters) * 100)) if total > 0 else 0

    return {
        "served": all_served,
        "total": total,
        "ctx_switches": ctx_switches,
        "ctx_penalty": ctx_penalty,
        "util": util,
        "timeline": merged_tl,
        "lanes": {
            "express": len(express),
            "standard": len(standard),
            "express_counters": express_counters,
            "standard_counters": standard_counters,
        },
    }


# ──────────────────────────────────────────────────────────────
# RUN ALL ALGORITHMS & PRINT COMPARISON
# ──────────────────────────────────────────────────────────────

def run_simulation(n_customers=10, n_counters=2, quantum=5, seed=42):
    """
    Run all five scheduling algorithms and return results dict.

    Parameters
    ----------
    n_customers : number of customers to generate
    n_counters  : number of checkout counters
    quantum     : RR / Multilevel time quantum
    seed        : RNG seed for reproducibility

    Returns
    -------
    dict keyed by algorithm name, each value containing simulation result
    and computed metrics.
    """
    customers = gen_customers(n_customers, seed)

    results = {
        "FCFS":       fcfs(customers, n_counters),
        "SJN":        sjn(customers, n_counters),
        "SRTF":       srtf(customers, n_counters),
        "RoundRobin": round_robin(customers, n_counters, quantum),
        "Multilevel": multilevel(customers, n_counters, quantum),
    }

    for name, res in results.items():
        res["metrics"] = calc_metrics(res)

    return customers, results


def print_report(customers, results):
    """Pretty-print a comparison table to stdout."""
    SEP = "─" * 72

    print("\n" + "═" * 72)
    print("  SUPERMARKET QMS — OS Scheduling Simulation")
    print("═" * 72)

    print(f"\n{'ID':>3}  {'Arrival':>8}  {'Items':>5}  {'Priority':>8}  {'Lane'}")
    print(SEP)
    for c in customers:
        pri = {1: "Regular", 2: "Senior", 3: "Staff"}[c["priority"]]
        print(f"{c['id']:>3}  {c['arrival']:>8.2f}  {c['items']:>5}  {pri:>8}  {c['lane']}")

    print("\n" + "═" * 72)
    print(f"  {'Algorithm':<14} {'AvgWait':>9} {'AvgTAT':>9} {'Throughput':>12} {'Util%':>7} {'CtxSw':>6}")
    print(SEP)

    for name, res in results.items():
        m = res["metrics"]
        print(f"  {name:<14} {m['avg_wait']:>9.2f} {m['avg_tat']:>9.2f} "
              f"{m['throughput']:>12.4f} {m['util']:>7.1f} {m['ctx']:>6}")

    print(SEP)

    for name, res in results.items():
        print(f"\n── {name} ── Customer Detail ──")
        print(f"  {'ID':>3}  {'Arrival':>8}  {'Items':>5}  {'Start':>8}  {'Finish':>8}  {'Wait':>7}  {'TAT':>7}")
        for c in sorted(res["served"], key=lambda x: x["id"]):
            wait = c["startTime"] - c["arrival"]
            tat  = c["finishTime"] - c["arrival"]
            print(f"  C{c['id']:>2}  {c['arrival']:>8.1f}  {c['items']:>5}  "
                  f"{c['startTime']:>8.1f}  {c['finishTime']:>8.1f}  "
                  f"{wait:>7.1f}  {tat:>7.1f}")
        print(f"  Context switches: {res['ctx_switches']}  "
              f"Penalty: +{res['ctx_penalty']:.1f} units  "
              f"Total duration: {res['total']:.1f} units")


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Default parameters — change these as needed
    N_CUSTOMERS = 10
    N_COUNTERS  = 2
    QUANTUM     = 5
    SEED        = 42

    customers, results = run_simulation(N_CUSTOMERS, N_COUNTERS, QUANTUM, SEED)
    print_report(customers, results)
