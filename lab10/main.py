import sys
import heapq
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QGroupBox,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox, QFrame,
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class EventType(Enum):
    ARRIVAL   = auto()
    DEPARTURE = auto()
    PATIENCE  = auto()

@dataclass(order=True)
class Event:
    time:       float
    etype:      EventType = field(compare=False)
    client_id:  int       = field(compare=False)


class Client:
    def __init__(self, cid: int, arrive_time: float,
                 patience: Optional[float] = None):
        self.id           = cid
        self.arrive_time  = arrive_time
        self.patience     = patience
        self.start_service: Optional[float] = None
        self.end_service:   Optional[float] = None
        self.reneged       = False


class Server:
    def __init__(self, sid: int):
        self.id      = sid
        self.busy    = False
        self.client: Optional[Client] = None

    def assign(self, client: Client, end_time: float):
        self.busy        = True
        self.client      = client
        client.start_service = end_time

    def free(self):
        self.busy   = False
        self.client = None


class MMnQueue:

    def __init__(self, lam: float, mu: float, n_servers: int,
                 queue_cap: Optional[int] = None,
                 patience_rate: Optional[float] = None):
        self.lam           = lam
        self.mu            = mu
        self.n_servers     = n_servers
        self.queue_cap     = queue_cap          # None - безлимит
        self.patience_rate = patience_rate      # None - бесконечно

        self.servers  = [Server(i) for i in range(n_servers)]
        self.queue:   list[Client] = []
        self.heap:    list[Event]  = []
        self.clients: dict[int, Client] = {}

        self.n_arrived  = 0
        self.n_served   = 0
        self.n_rejected = 0
        self.n_reneged  = 0 
        self.wait_times: list[float] = []
        self.system_times: list[float] = []


    def _push(self, e: Event):
        heapq.heappush(self.heap, e)

    def _pop(self) -> Event:
        return heapq.heappop(self.heap)

    def _free_server(self) -> Optional[Server]:
        for s in self.servers:
            if not s.busy:
                return s
        return None


    def run(self, n_clients: int):
        t = 0.0
        next_id = 0

        t_arrive = np.random.exponential(1.0 / self.lam)
        self._push(Event(t_arrive, EventType.ARRIVAL, next_id))

        while self.heap:
            ev = self._pop()
            t  = ev.time

            if ev.etype == EventType.ARRIVAL:
                self._handle_arrival(ev, t, next_id, n_clients)
                next_id += 1
                if next_id < n_clients:
                    t_next = t + np.random.exponential(1.0 / self.lam)
                    self._push(Event(t_next, EventType.ARRIVAL, next_id))

            elif ev.etype == EventType.DEPARTURE:
                self._handle_departure(ev, t)

            elif ev.etype == EventType.PATIENCE:
                self._handle_patience(ev, t)

        return self._collect_stats()

    def _handle_arrival(self, ev: Event, t: float, cid: int, n_total: int):
        self.n_arrived += 1

        patience = (
            np.random.exponential(1.0 / self.patience_rate)
            if self.patience_rate else None
        )
        client = Client(cid, t, patience)
        self.clients[cid] = client

        srv = self._free_server()
        if srv:
            self._start_service(client, srv, t)
        else:
            if self.queue_cap is not None and len(self.queue) >= self.queue_cap:
                self.n_rejected += 1
                return
            # enqueue
            self.queue.append(client)
            if patience is not None:
                deadline = t + patience
                self._push(Event(deadline, EventType.PATIENCE, cid))

    def _start_service(self, client: Client, srv: Server, t: float):
        service_time = np.random.exponential(1.0 / self.mu)
        client.start_service = t
        srv.busy   = True
        srv.client = client
        end_t = t + service_time
        client.end_service = end_t
        self._push(Event(end_t, EventType.DEPARTURE, client.id))

    def _handle_departure(self, ev: Event, t: float):
        client = self.clients.get(ev.client_id)
        if client is None or client.reneged:
            return

        srv = next((s for s in self.servers if s.client is client), None)
        if srv is None:
            return

        self.n_served += 1
        wait = client.start_service - client.arrive_time
        sys_t = t - client.arrive_time
        self.wait_times.append(wait)
        self.system_times.append(sys_t)
        srv.free()

        if self.queue:
            next_client = self.queue.pop(0)
            self._start_service(next_client, srv, t)

    def _handle_patience(self, ev: Event, t: float):
        client = self.clients.get(ev.client_id)
        if client is None or client.start_service is not None:
            return
        if client in self.queue:
            self.queue.remove(client)
            client.reneged = True
            self.n_reneged += 1

    def _collect_stats(self) -> dict:
        wt = np.array(self.wait_times)
        st = np.array(self.system_times)
        return {
            "n_arrived":  self.n_arrived,
            "n_served":   self.n_served,
            "n_rejected": self.n_rejected,
            "n_reneged":  self.n_reneged,
            "mean_wait":  np.mean(wt) if len(wt) else 0.0,
            "mean_sys":   np.mean(st) if len(st) else 0.0,
            "p_reject":   self.n_rejected / self.n_arrived if self.n_arrived else 0,
            "p_renege":   self.n_reneged  / self.n_arrived if self.n_arrived else 0,
            "utilization": self.n_served / (self.n_servers * (
                self.n_arrived / self.lam
            )) if self.n_arrived else 0,
            "wait_times":  wt,
            "system_times": st,
        }


class MplCanvas(FigureCanvas):
    def __init__(self, figsize=(7, 3.5)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.ax  = self.fig.add_subplot(111)
        super().__init__(self.fig)


class TwoAxCanvas(FigureCanvas):
    def __init__(self, figsize=(9, 3.8)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.ax1, self.ax2 = self.fig.subplots(1, 2)
        super().__init__(self.fig)


class Divider(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setStyleSheet("color: #d0d8e8;")


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лаб. 10 — СМО M/M/n с ограничениями")
        self.resize(1280, 800)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl_box = QGroupBox("Параметры системы")
        ctrl_layout = QGridLayout(ctrl_box)
        ctrl_layout.setHorizontalSpacing(12)
        ctrl_layout.setVerticalSpacing(6)

        def field(label, default, w=80):
            lbl = QLabel(label)
            inp = QLineEdit(default)
            inp.setFixedWidth(w)
            inp.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return lbl, inp

        lbl_lam, self.inp_lam = field("λ:", "5.0")
        lbl_mu,  self.inp_mu  = field("μ:", "3.0")
        lbl_n,   self.inp_n   = field("Приборов (n):", "3", 60)
        lbl_nc,  self.inp_nc  = field("Заявок:", "5000", 80)

        ctrl_layout.addWidget(lbl_lam,  0, 0); ctrl_layout.addWidget(self.inp_lam, 0, 1)
        ctrl_layout.addWidget(lbl_mu,   0, 2); ctrl_layout.addWidget(self.inp_mu,  0, 3)
        ctrl_layout.addWidget(lbl_n,    0, 4); ctrl_layout.addWidget(self.inp_n,   0, 5)
        ctrl_layout.addWidget(lbl_nc,   0, 6); ctrl_layout.addWidget(self.inp_nc,  0, 7)

        self.chk_cap = QCheckBox("Ограничение очереди")
        self.chk_cap.setChecked(True)
        lbl_cap, self.inp_cap = field("Размер буфера:", "10", 70)
        self.chk_cap.toggled.connect(lambda v: self.inp_cap.setEnabled(v))

        self.chk_pat = QCheckBox("Нетерпеливость заявок")
        self.chk_pat.setChecked(True)
        lbl_pat, self.inp_pat = field("γ (интенсивность ухода):", "0.5", 70)
        self.chk_pat.toggled.connect(lambda v: self.inp_pat.setEnabled(v))

        ctrl_layout.addWidget(self.chk_cap, 1, 0, 1, 2)
        ctrl_layout.addWidget(lbl_cap,      1, 2); ctrl_layout.addWidget(self.inp_cap, 1, 3)
        ctrl_layout.addWidget(self.chk_pat, 1, 4, 1, 2)
        ctrl_layout.addWidget(lbl_pat,      1, 6); ctrl_layout.addWidget(self.inp_pat, 1, 7)

        self.run_btn = QPushButton("Моделировать")
        self.run_btn.setFixedWidth(140)
        self.run_btn.clicked.connect(self.run)
        ctrl_layout.addWidget(self.run_btn, 0, 8, 2, 1, Qt.AlignmentFlag.AlignVCenter)

        root.addWidget(ctrl_box)

        split = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        dist_box = QGroupBox("Гистограммы времён")
        dist_layout = QVBoxLayout(dist_box)
        self.hist_canvas = TwoAxCanvas(figsize=(9, 3.8))
        dist_layout.addWidget(self.hist_canvas)
        ll.addWidget(dist_box)

        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        stats_box = QGroupBox("Статистика моделирования")
        sl = QVBoxLayout(stats_box)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Показатель", "Значение"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        sl.addWidget(self.table)

        rl.addWidget(stats_box)

        self.conclusion_lbl = QLabel("")
        self.conclusion_lbl.setWordWrap(True)
        self.conclusion_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.conclusion_lbl.setStyleSheet(
            "background: white; border: 1px solid #d0d8e8; border-radius: 8px; padding: 10px;"
        )
        rl.addWidget(self.conclusion_lbl)
        rl.addStretch()

        split.addWidget(right)
        split.setSizes([780, 420])
        root.addWidget(split, stretch=1)

        self.apply_style()

    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f4f6fb; font-size: 13px; }
            QGroupBox {
                border: 1px solid #d0d8e8; border-radius: 8px;
                margin-top: 10px; background: white; font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
            QPushButton {
                background: #2d6cdf; color: white; border: none;
                padding: 8px 14px; border-radius: 6px;
            }
            QPushButton:hover   { background: #2458b8; }
            QPushButton:pressed { background: #1f4b9c; }
            QLineEdit {
                padding: 5px 8px; border: 1px solid #cfd8e3;
                border-radius: 6px; background: white;
            }
            QLineEdit:disabled { background: #f0f4fa; color: #aaa; }
            QCheckBox { font-weight: normal; }
            QTableWidget {
                background: white; border: 1px solid #cfd8e3;
                border-radius: 6px; gridline-color: #e0e6ef;
            }
            QHeaderView::section {
                background: #eef3f9; padding: 6px; border: none;
                border-bottom: 1px solid #d7dee8; font-weight: bold;
            }
            QTableWidget::item:alternate { background: #f8fafc; }
        """)

    def run(self):
        try:
            lam = float(self.inp_lam.text().replace(",", "."))
            mu  = float(self.inp_mu.text().replace(",", "."))
            n   = int(self.inp_n.text())
            nc  = int(self.inp_nc.text())
            assert lam > 0 and mu > 0 and n >= 1 and nc > 10

            queue_cap = int(self.inp_cap.text()) if self.chk_cap.isChecked() else None
            pat_rate  = float(self.inp_pat.text().replace(",", ".")) if self.chk_pat.isChecked() else None
            if queue_cap is not None:
                assert queue_cap >= 0
            if pat_rate is not None:
                assert pat_rate > 0
        except Exception:
            self.conclusion_lbl.setText("Ошибка в параметрах.")
            return

        sim = MMnQueue(lam, mu, n, queue_cap=queue_cap, patience_rate=pat_rate)
        stats = sim.run(nc)

        self._fill_table(stats, lam, mu, n)
        self._draw_plots(stats)
        self._write_conclusion(stats, lam, mu, n)

    def _fill_table(self, s: dict, lam, mu, n):
        rho = lam / (n * mu)
        rows = [
            ("Прибыло заявок",             f"{s['n_arrived']}"),
            ("Обслужено",                  f"{s['n_served']}"),
            ("Отказано (переполнение)",     f"{s['n_rejected']}"),
            ("Покинули очередь сами",       f"{s['n_reneged']}"),
            ("P(отказ)",                   f"{s['p_reject']:.3%}"),
            ("P(уйти из очереди)",         f"{s['p_renege']:.3%}"),
            ("Среднее время ожидания Wq",   f"{s['mean_wait']:.4f}"),
            ("Среднее время в системе W",   f"{s['mean_sys']:.4f}"),
            ("ρ (предложенная нагрузка)",   f"{rho:.3f}"),
        ]
        self.table.setRowCount(len(rows))
        for i, (name, val) in enumerate(rows):
            n_item = QTableWidgetItem(name)
            v_item = QTableWidgetItem(val)
            v_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 0, n_item)
            self.table.setItem(i, 1, v_item)

    def _draw_plots(self, s: dict):
        ax1, ax2 = self.hist_canvas.ax1, self.hist_canvas.ax2
        ax1.clear(); ax2.clear()

        wt = s["wait_times"]
        st = s["system_times"]

        if len(wt) > 0:
            ax1.hist(wt, bins=50, density=True, alpha=0.75,
                     color="#3b82f6", edgecolor="white")
            ax1.set_xlabel("Время ожидания")
            ax1.set_ylabel("Плотность")
            ax1.set_title("Распределение времени ожидания")
            ax1.grid(True, alpha=0.25)

        if len(st) > 0:
            ax2.hist(st, bins=50, density=True, alpha=0.75,
                     color="#10b981", edgecolor="white")
            ax2.set_xlabel("Время в системе")
            ax2.set_ylabel("Плотность")
            ax2.set_title("Распределение времени в системе")
            ax2.grid(True, alpha=0.25)

        self.hist_canvas.draw()

    def _write_conclusion(self, s: dict, lam, mu, n):
        lines = []
        rho = lam / (n * mu)
        lines.append(f"Приборов: {n}, ρ = λ/(nμ) = {rho:.3f}")
        if rho < 1:
            lines.append("Система устойчива (ρ < 1).")
        else:
            lines.append("Внимание: ρ ≥ 1 — без ограничений очередь бы росла.")

        if s["n_rejected"]:
            lines.append(f"Отказано {s['n_rejected']} заявкам ({s['p_reject']:.1%}) "
                         "из-за переполнения буфера.")
        if s["n_reneged"]:
            lines.append(f"{s['n_reneged']} заявок ({s['p_renege']:.1%}) "
                         "покинули очередь из-за нетерпеливости.")

        lines.append(f"Среднее время ожидания: {s['mean_wait']:.4f} ед.")
        lines.append(f"Среднее время в системе: {s['mean_sys']:.4f} ед.")

        self.conclusion_lbl.setText("\n".join(lines))

def main():
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
