import argparse
import heapq
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field


class EventType(Enum):
    ARRIVAL = auto()
    DEPARTURE = auto()
    BREAKDOWN = auto()
    REPAIR = auto()


@dataclass(order=True)
class Event:
    time: float
    etype: EventType = field(compare=False)


class MM1LossSimulator:
    """M/M/1/0 — одноканальная СМО без очереди."""

    def __init__(self, lam: float, mu: float,
                 breakdown_interval: float = 200.0,
                 repair_duration: float = 20.0):
        self.lam = lam
        self.mu = mu
        self.breakdown_interval = breakdown_interval
        self.repair_duration = repair_duration

    def run(self, sim_time: float) -> dict:
        lam, mu = self.lam, self.mu
        heap: list[Event] = []
        busy = False
        broken = False  # прибор сломан
        pending_breakdown = False  # поломка ожидает окончания обслуживания

        n_arrived = 0
        n_served = 0
        n_lost = 0
        n_breakdowns = 0
        busy_time = 0.0
        down_time = 0.0
        service_times: list[float] = []

        t_arrive = np.random.exponential(1.0 / lam)
        heapq.heappush(heap, Event(t_arrive, EventType.ARRIVAL))

        heapq.heappush(heap, Event(self.breakdown_interval, EventType.BREAKDOWN))

        last_busy_start = 0.0

        while heap:
            ev = heapq.heappop(heap)
            t = ev.time
            if t > sim_time:
                if busy and not broken:
                    busy_time += sim_time - last_busy_start
                break

            if ev.etype == EventType.ARRIVAL:
                n_arrived += 1
                if not busy and not broken:
                    busy = True
                    last_busy_start = t
                    svc = np.random.exponential(1.0 / mu)
                    service_times.append(svc)
                    heapq.heappush(heap, Event(t + svc, EventType.DEPARTURE))
                else:
                    n_lost += 1

                t_next = t + np.random.exponential(1.0 / lam)
                if t_next <= sim_time:
                    heapq.heappush(heap, Event(t_next, EventType.ARRIVAL))

            elif ev.etype == EventType.DEPARTURE:
                n_served += 1
                busy_time += t - last_busy_start
                busy = False
                if pending_breakdown:
                    pending_breakdown = False
                    broken = True
                    n_breakdowns += 1
                    heapq.heappush(heap, Event(t + self.repair_duration, EventType.REPAIR))

            elif ev.etype == EventType.BREAKDOWN:
                if busy:
                    # откладываем поломку до конца обслуживания
                    pending_breakdown = True
                else:
                    n_breakdowns += 1
                    broken = True
                    heapq.heappush(heap, Event(t + self.repair_duration, EventType.REPAIR))

            elif ev.etype == EventType.REPAIR:
                down_time += self.repair_duration
                broken = False

                # следующая поломка через breakdown_interval
                next_bd = t + self.breakdown_interval
                if next_bd <= sim_time:
                    heapq.heappush(heap, Event(next_bd, EventType.BREAKDOWN))

        return {
            "n_arrived": n_arrived,
            "n_served": n_served,
            "n_lost": n_lost,
            "n_breakdowns": n_breakdowns,
            "busy_time": busy_time,
            "down_time": down_time,
            "sim_time": sim_time,
            "service_times": np.array(service_times),
        }


def theoretical(lam: float, mu: float,
                breakdown_interval: float = 200.0,
                repair_duration: float = 20.0) -> dict:
    """Теоретические характеристики M/M/1/0 с поломками."""
    # доля времени, когда прибор доступен (не сломан)
    avail = breakdown_interval / (breakdown_interval + repair_duration)

    rho = lam / mu
    p0_pure = 1.0 / (1.0 + rho) # вероятность что прибор свободен без учета поломок
    p1_pure = rho / (1.0 + rho) # вероятность что прибор занят без учета поломок

    # с учётом поломок: прибор недоступен долю (1 - avail) времени
    # P(потери) = P(занят|доступен)*avail + P(сломан)
    p_down = 1 - avail
    p_busy = p1_pure * avail
    p_free = p0_pure * avail
    p_loss = p_busy + p_down  # заявка теряется если занят или сломан
    throughput = lam * (1 - p_loss)
    L = p_busy
    W = 1.0 / mu

    return {
        "rho": rho,
        "avail": avail,
        "P_free": p_free,
        "P_busy": p_busy,
        "P_down": p_down,
        "P_loss": p_loss,
        "throughput": throughput,
        "L": L,
        "W": W,
    }


def main():
    parser = argparse.ArgumentParser(
        description="M/M/1/0 — одноканальная СМО с потерями (без очереди)"
    )
    parser.add_argument("--lam", type=float, default=4.0,
                        help="интенсивность поступления заявок λ (по умолчанию 4.0)")
    parser.add_argument("--mu", type=float, default=6.0,
                        help="интенсивность обслуживания μ (по умолчанию 6.0)")
    parser.add_argument("--time", type=float, default=10000.0,
                        help="модельное время (по умолчанию 10000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed для воспроизводимости")
    args = parser.parse_args()

    if args.lam <= 0 or args.mu <= 0 or args.time <= 0:
        print("Ошибка: λ, μ и время моделирования должны быть > 0")
        return

    if args.seed is not None:
        np.random.seed(args.seed)

    lam, mu, sim_time = args.lam, args.mu, args.time
    bd_interval = 200.0
    repair_dur = 20.0

    print("=" * 60)
    print("  M/M/1/0 — СМО без очереди + поломки")
    print("=" * 60)
    print(f"  λ = {lam},  μ = {mu},  T = {sim_time}")
    print(f"  ρ = λ/μ = {lam / mu:.4f}")
    print(f"  Поломка каждые {bd_interval} ед., ремонт {repair_dur} ед.")
    print()

    # теория
    th = theoretical(lam, mu, bd_interval, repair_dur)
    print("─── Теоретические значения ───")
    print(f"  Доступность прибора       = {th['avail']:.6f}")
    print(f"  P(свободен)               = {th['P_free']:.6f}")
    print(f"  P(занят)                  = {th['P_busy']:.6f}")
    print(f"  P(сломан)                 = {th['P_down']:.6f}")
    print(f"  P_потери                  = {th['P_loss']:.6f}")
    print()

    # моделирование
    sim = MM1LossSimulator(lam, mu, bd_interval, repair_dur)
    res = sim.run(sim_time)

    p_loss_emp = res["n_lost"] / res["n_arrived"] if res["n_arrived"] else 0
    utilization = res["busy_time"] / sim_time
    down_frac = res["down_time"] / sim_time
    free_frac = 1 - utilization - down_frac
    throughput_emp = res["n_served"] / sim_time
    L_emp = utilization
    W_emp = np.mean(res["service_times"]) if len(res["service_times"]) else 0

    print("─── Результаты моделирования ───")
    print(f"  Прибыло заявок            = {res['n_arrived']}")
    print(f"  Обслужено                 = {res['n_served']}")
    print(f"  Потеряно                  = {res['n_lost']}")
    print(f"  Число поломок             = {res['n_breakdowns']}")
    print(f"  P(свободен)               = {free_frac:.6f}")
    print(f"  P(занят)                  = {utilization:.6f}")
    print(f"  P(сломан)                 = {down_frac:.6f}")
    print(f"  P_потери                  = {p_loss_emp:.6f}")
    print()

    # сравнение
    print("─── Сравнение (теория vs эмпирика) ───")
    print(f"  {'Показатель':<28} {'Теория':>10} {'Эмпирика':>10} {'Δ':>10}")
    print(f"  {'─' * 58}")
    comparisons = [
        ("P(свободен)", th["P_free"], free_frac),
        ("P(занят)", th["P_busy"], utilization),
        ("P(сломан)", th["P_down"], down_frac),
        ("P_потери", th["P_loss"], p_loss_emp),
    ]
    for name, tv, ev in comparisons:
        delta = abs(tv - ev)
        print(f"  {name:<28} {tv:>10.6f} {ev:>10.6f} {delta:>10.6f}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
