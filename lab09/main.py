import sys
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def simulate_mm1(lam: float, mu: float, n_clients: int):
    arrival_times = np.cumsum(np.random.exponential(1.0 / lam, n_clients))
    service_times = np.random.exponential(1.0 / mu, n_clients)

    start_service = np.zeros(n_clients)
    end_service   = np.zeros(n_clients)
    wait_times    = np.zeros(n_clients)

    server_free_at = 0.0
    for i in range(n_clients):
        start = max(arrival_times[i], server_free_at)
        start_service[i] = start
        end_service[i]   = start + service_times[i]
        server_free_at   = end_service[i]
        wait_times[i]    = start - arrival_times[i]


    queue_lengths = np.zeros(n_clients, dtype=int)
    for i in range(n_clients):
        in_system = int(np.sum(
            (arrival_times[:i] <= arrival_times[i]) &
            (end_service[:i]   > arrival_times[i])
        ))
        queue_lengths[i] = in_system

    return {
        "arrival_times": arrival_times,
        "start_service": start_service,
        "end_service":   end_service,
        "wait_times":    wait_times,
        "queue_lengths": queue_lengths,
        "service_times": service_times,
    }


def mm1_theory(lam: float, mu: float):
    rho = lam / mu
    if rho >= 1:
        return None
    return {
        "rho": rho,
        "L":   rho / (1 - rho),
        "Lq":  rho ** 2 / (1 - rho),
        "W":   1.0 / (mu - lam),
        "Wq":  rho / (mu - lam),
        "lam": lam,
        "mu":  mu,
    }


# ── Canvas helpers ────────────────────────────────────────────────────────────

class MplCanvas(FigureCanvas):
    def __init__(self, figsize=(6, 3.5)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.ax  = self.fig.add_subplot(111)
        super().__init__(self.fig)


class TwoAxCanvas(FigureCanvas):
    def __init__(self, figsize=(9, 3.8)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.ax1, self.ax2 = self.fig.subplots(1, 2)
        super().__init__(self.fig)


# ── Main window ───────────────────────────────────────────────────────────────

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("M/M/1 — Имитационное моделирование")
        self.resize(1260, 760)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl_box = QGroupBox("Параметры системы")
        ctrl_layout = QHBoxLayout(ctrl_box)

        def field(label, default, w=90):
            lbl = QLabel(label)
            inp = QLineEdit(default)
            inp.setFixedWidth(w)
            inp.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return lbl, inp

        lbl_lam, self.inp_lam = field("λ (интенс. поступления):", "4.0")
        lbl_mu,  self.inp_mu  = field("μ (интенс. обслуживания):", "6.0")
        lbl_n,   self.inp_n   = field("N (число заявок):", "5000")

        for w in (lbl_lam, self.inp_lam, lbl_mu, self.inp_mu, lbl_n, self.inp_n):
            ctrl_layout.addWidget(w)

        self.run_btn = QPushButton("Моделировать")
        self.run_btn.setFixedWidth(130)
        self.run_btn.clicked.connect(self.run)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.run_btn)
        root.addWidget(ctrl_box)

        # ── Main area ─────────────────────────────────────────────────────────
        split = QSplitter(Qt.Orientation.Horizontal)

        # Left: plots
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        distr_box = QGroupBox("Распределения: время ожидания и длина очереди")
        distr_layout = QVBoxLayout(distr_box)
        self.two_canvas = TwoAxCanvas(figsize=(9, 3.8))
        distr_layout.addWidget(self.two_canvas)
        left_layout.addWidget(distr_box)

        dyn_box = QGroupBox("Динамика длины очереди (первые 200 заявок)")
        dyn_layout = QVBoxLayout(dyn_box)
        self.queue_canvas = MplCanvas(figsize=(9, 2.8))
        dyn_layout.addWidget(self.queue_canvas)
        left_layout.addWidget(dyn_box)

        split.addWidget(left)

        # Right: table + conclusion
        stats_box = QGroupBox("Сравнение: эмпирика vs теория")
        stats_layout = QVBoxLayout(stats_box)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Показатель", "Эмпирика", "Теория"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        stats_layout.addWidget(self.table)

        self.conclusion_lbl = QLabel("")
        self.conclusion_lbl.setWordWrap(True)
        self.conclusion_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        stats_layout.addWidget(self.conclusion_lbl)

        split.addWidget(stats_box)
        split.setSizes([760, 380])
        root.addWidget(split, stretch=1)

        self.apply_style()

    # ── Style ─────────────────────────────────────────────────────────────────

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

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        try:
            lam = float(self.inp_lam.text().replace(",", "."))
            mu  = float(self.inp_mu.text().replace(",", "."))
            n   = int(self.inp_n.text())
            assert lam > 0 and mu > 0 and n > 10
        except Exception:
            self.conclusion_lbl.setText("Ошибка ввода.")
            return

        theory = mm1_theory(lam, mu)
        if theory is None:
            self.conclusion_lbl.setText(
                "Система неустойчива (ρ ≥ 1): очередь уходит в бесконечность.\n"
                "Увеличьте μ или уменьшите λ."
            )
            self.conclusion_lbl.setStyleSheet("color: #991b1b; font-weight: bold;")
            return

        res = simulate_mm1(lam, mu, n)

        wait = res["wait_times"]
        ql   = res["queue_lengths"]
        srv  = res["service_times"]

        emp_Wq  = np.mean(wait)
        emp_W   = np.mean(wait + srv)
        emp_Lq  = np.mean(ql)
        emp_L   = lam * emp_W 
        emp_rho = lam * np.mean(srv) 

        self._fill_table(theory, {
            "rho": emp_rho, "L": emp_L,
            "Lq":  emp_Lq,  "W": emp_W, "Wq": emp_Wq,
        })
        self._draw_plots(res, theory)

        rho = theory["rho"]
        self.conclusion_lbl.setStyleSheet("color: #166534; font-weight: normal;")
        self.conclusion_lbl.setText(
            f"ρ = {rho:.3f} < 1 — система устойчива.\n"
            f"Загрузка прибора: {rho*100:.1f} %.\n"
            f"Доля заявок без ожидания: {np.mean(wait == 0)*100:.1f} % "
            f"(теория: {(1-rho)*100:.1f} %).\n"
            f"Среднее время ожидания Wq: {emp_Wq:.4f}  (теория: {theory['Wq']:.4f}).\n"
            f"Среднее число в очереди Lq: {emp_Lq:.4f}  (теория: {theory['Lq']:.4f})."
        )

    # ── Table ─────────────────────────────────────────────────────────────────

    def _fill_table(self, theory, empiric):
        rows = [
            ("ρ (коэффициент загрузки)",    "rho"),
            ("L (среднее число в системе)",  "L"),
            ("Lq (среднее число в очереди)", "Lq"),
            ("W (среднее время в системе)",  "W"),
            ("Wq (среднее время ожидания)",  "Wq"),
        ]
        self.table.setRowCount(len(rows))
        for i, (name, key) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(name))
            for col, d in enumerate((empiric, theory), start=1):
                item = QTableWidgetItem(f"{d[key]:.4f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(i, col, item)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _draw_plots(self, res, theory):
        rho = theory["rho"]
        lam = theory["lam"]
        mu  = theory["mu"]
        rate = mu - lam          # = μ(1−ρ), параметр экспоненциального хвоста Wq

        wait = res["wait_times"]
        ql   = res["queue_lengths"]

        ax1, ax2 = self.two_canvas.ax1, self.two_canvas.ax2
        ax1.clear(); ax2.clear()

        # ── График 1: распределение Wq ────────────────────────────────────────
        # Полная гистограмма (включая нули).
        # Теоретическая плотность непрерывной части: ρ(μ−λ)·exp(−(μ−λ)t) для t>0
        t_max = np.percentile(wait[wait > 1e-10], 99) if np.any(wait > 1e-10) else 1.0
        ax1.hist(wait, bins=40, density=True, alpha=0.65,
                 color="#3b82f6", edgecolor="white", label="Эмпирика")
        xs = np.linspace(1e-6, t_max, 400)
        ax1.plot(xs, rho * rate * np.exp(-rate * xs),
                 color="#ef4444", linewidth=2.2,
                 label=f"Теория: ρ(μ−λ)e^{{−(μ−λ)t}}")
        ax1.set_xlabel("Время ожидания Wq")
        ax1.set_ylabel("Плотность")
        ax1.set_title("Время ожидания в очереди")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.25)

        # ── График 2: распределение L (число в системе) ──────────────────────
        # Теория M/M/1: P(L=n) = (1−ρ)·ρ^n, n = 0,1,2,...
        k_max = min(int(ql.max()), 25) if ql.max() > 0 else 5
        bins = np.arange(-0.5, k_max + 1.5, 1)
        ax2.hist(ql, bins=bins, density=True, alpha=0.65,
                 color="#8b5cf6", edgecolor="white", label="Эмпирика")

        ks = np.arange(0, k_max + 1)
        th_ql = ((1 - rho) * rho ** ks).astype(float)   # геометрическое распределение
        ax2.plot(ks, th_ql, "o-", color="#ef4444",
                 markersize=5, linewidth=2, label="Теория")
        ax2.set_xlabel("Число заявок в системе L")
        ax2.set_ylabel("Вероятность")
        ax2.set_title("Число заявок в системе (L)")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.25)

        self.two_canvas.draw()

        # ── График 3: динамика очереди ────────────────────────────────────────
        ax = self.queue_canvas.ax
        ax.clear()
        n_show = min(200, len(res["arrival_times"]))
        t = res["arrival_times"][:n_show]
        q = res["queue_lengths"][:n_show]
        ax.step(t, q, where="post", color="#2d6cdf", linewidth=1.4)
        ax.axhline(theory["Lq"], color="#ef4444", linewidth=1.4,
                   linestyle="--", label=f"Теор. Lq = {theory['Lq']:.3f}")
        ax.set_xlabel("Время прибытия заявки")
        ax.set_ylabel("Lq")
        ax.set_title("Динамика длины очереди (первые 200 заявок)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        self.queue_canvas.draw()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
