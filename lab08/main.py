import sys
import numpy as np
from scipy.stats import poisson, chisquare, chi2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QTextEdit,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def simulate_poisson_flow(lam: float, T: float, n_runs: int) -> np.ndarray:
    counts = np.zeros(n_runs, dtype=int)
    for i in range(n_runs):
        t = 0.0
        k = 0
        while True:
            dt = np.random.exponential(1.0 / lam)
            if t + dt > T:
                break
            t += dt
            k += 1
        counts[i] = k
    return counts


def analyze(counts: np.ndarray, lam: float, T: float):
    n = len(counts)
    theoretical_mean = lam * T
    theoretical_var  = lam * T

    emp_mean = np.mean(counts)
    emp_var  = np.var(counts, ddof=1)

    k_min, k_max = counts.min(), counts.max()
    obs_vals = np.arange(k_min, k_max + 1)
    observed = np.array([np.sum(counts == k) for k in obs_vals], dtype=float)
    expected = np.array([n * poisson.pmf(k, lam * T) for k in obs_vals], dtype=float)

    while len(expected) > 1 and expected[0] < 5:
        observed[1] += observed[0]; observed = observed[1:]
        expected[1] += expected[0]; expected = expected[1:]
        obs_vals = obs_vals[1:]
    while len(expected) > 1 and expected[-1] < 5:
        observed[-2] += observed[-1]; observed = observed[:-1]
        expected[-2] += expected[-1]; expected = expected[:-1]
        obs_vals = obs_vals[:-1]

    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = len(observed) - 1
    chi2_crit = chi2.ppf(0.95, df)
    p_value   = 1 - chi2.cdf(chi2_stat, df)

    return emp_mean, emp_var, theoretical_mean, theoretical_var, chi2_stat, chi2_crit, p_value

class MplCanvas(FigureCanvas):
    def __init__(self, figsize=(8, 4)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.ax  = self.fig.add_subplot(111)
        super().__init__(self.fig)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Пуассоновский поток")
        self.resize(1100, 720)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)

        ctrl_box = QGroupBox("Параметры")
        ctrl_layout = QHBoxLayout(ctrl_box)

        def labeled(text, default, width=90):
            lbl = QLabel(text)
            inp = QLineEdit(default)
            inp.setFixedWidth(width)
            inp.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return lbl, inp

        lbl_lam, self.inp_lam   = labeled("λ (интенсивность):", "5.0")
        lbl_T,   self.inp_T     = labeled("T (интервал):", "10.0")
        lbl_n,   self.inp_n     = labeled("N (прогонов):", "2000")

        for w in (lbl_lam, self.inp_lam, lbl_T, self.inp_T, lbl_n, self.inp_n):
            ctrl_layout.addWidget(w)

        self.run_btn = QPushButton("Запустить")
        self.run_btn.setFixedWidth(120)
        self.run_btn.clicked.connect(self.run)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.run_btn)

        root.addWidget(ctrl_box)

        split = QSplitter(Qt.Orientation.Horizontal)

        plot_box = QGroupBox("Распределение числа событий")
        plot_layout = QVBoxLayout(plot_box)
        self.canvas = MplCanvas(figsize=(7, 4))
        plot_layout.addWidget(self.canvas)
        split.addWidget(plot_box)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        stats_box = QGroupBox("Статистика")
        stats_layout = QGridLayout(stats_box)
        stats_layout.setColumnMinimumWidth(1, 110)

        self._stat_labels = {}
        rows = [
            ("emp_mean",   "Выборочное среднее:"),
            ("emp_var",    "Выборочная дисперсия:"),
            ("th_mean",    "Теоретическое среднее (λT):"),
            ("th_var",     "Теоретическая дисперсия (λT):"),
            ("err_mean",   "Отн. погрешность среднего:"),
            ("err_var",    "Отн. погрешность дисперсии:"),
            ("chi2_stat",  "χ²:"),
            ("chi2_crit",  "χ² критическое (0.95):"),
            ("p_value",    "p-value:"),
            ("conclusion", "Вывод:"),
        ]
        for r, (key, text) in enumerate(rows):
            stats_layout.addWidget(QLabel(text), r, 0, Qt.AlignmentFlag.AlignLeft)
            val_lbl = QLabel("—")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._stat_labels[key] = val_lbl
            stats_layout.addWidget(val_lbl, r, 1)

        right_layout.addWidget(stats_box)

        timeline_box = QGroupBox("Пример реализации потока (первые 20 с)")
        tl_layout = QVBoxLayout(timeline_box)
        self.timeline_canvas = MplCanvas(figsize=(5, 2))
        tl_layout.addWidget(self.timeline_canvas)
        right_layout.addWidget(timeline_box)

        right_layout.addStretch()
        split.addWidget(right)
        split.setSizes([620, 420])

        root.addWidget(split, stretch=1)

        self.apply_style()

    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f4f6fb; font-size: 13px; }
            QGroupBox {
                border: 1px solid #d0d8e8;
                border-radius: 8px;
                margin-top: 10px;
                background: white;
                font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
            QPushButton {
                background: #2d6cdf; color: white; border: none;
                padding: 8px 14px; border-radius: 6px;
            }
            QPushButton:hover  { background: #2458b8; }
            QPushButton:pressed { background: #1f4b9c; }
            QLineEdit {
                padding: 5px 8px; border: 1px solid #cfd8e3;
                border-radius: 6px; background: white;
            }
        """)

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self):
        try:
            lam = float(self.inp_lam.text().replace(",", "."))
            T   = float(self.inp_T.text().replace(",", "."))
            n   = int(self.inp_n.text())
            assert lam > 0 and T > 0 and n > 0
        except Exception:
            self._stat_labels["conclusion"].setText("Ошибка ввода")
            return

        counts = simulate_poisson_flow(lam, T, n)
        emp_mean, emp_var, th_mean, th_var, chi2_stat, chi2_crit, p_value = analyze(counts, lam, T)

        err_mean = abs(emp_mean - th_mean) / th_mean if th_mean else 0
        err_var  = abs(emp_var  - th_var)  / th_var  if th_var  else 0

        self._stat_labels["emp_mean"].setText(f"{emp_mean:.4f}")
        self._stat_labels["emp_var"].setText(f"{emp_var:.4f}")
        self._stat_labels["th_mean"].setText(f"{th_mean:.4f}")
        self._stat_labels["th_var"].setText(f"{th_var:.4f}")
        self._stat_labels["err_mean"].setText(f"{err_mean:.2%}")
        self._stat_labels["err_var"].setText(f"{err_var:.2%}")
        self._stat_labels["chi2_stat"].setText(f"{chi2_stat:.4f}")
        self._stat_labels["chi2_crit"].setText(f"{chi2_crit:.4f}")
        self._stat_labels["p_value"].setText(f"{p_value:.4f}")

        accepted = chi2_stat <= chi2_crit
        text = "Гипотеза ПРИНЯТА" if accepted else "Гипотеза ОТВЕРГНУТА"
        color = "#166534" if accepted else "#991b1b"
        self._stat_labels["conclusion"].setText(text)
        self._stat_labels["conclusion"].setStyleSheet(f"color: {color}; font-weight: bold;")

        self._draw_histogram(counts, lam, T)
        self._draw_timeline(lam, T)

    def _draw_histogram(self, counts, lam, T):
        ax = self.canvas.ax
        ax.clear()

        k_min = max(0, counts.min())
        k_max = counts.max()
        bins  = np.arange(k_min - 0.5, k_max + 1.5, 1)
        ax.hist(counts, bins=bins, density=True, alpha=0.65, color="#3b82f6",
                edgecolor="white", label="Эмпирическое")

        ks = np.arange(k_min, k_max + 1)
        pmf = poisson.pmf(ks, lam * T)
        ax.plot(ks, pmf, "o-", color="#ef4444", linewidth=1.8,
                markersize=5, label=f"Пуассон(λT={lam*T:.1f})")

        ax.set_xlabel("Число событий за T")
        ax.set_ylabel("Вероятность")
        ax.set_title("Эмпирическое vs Теоретическое распределение")
        ax.legend()
        ax.grid(True, alpha=0.25)
        self.canvas.draw()

    def _draw_timeline(self, lam, T_show=20.0):
        ax = self.timeline_canvas.ax
        ax.clear()

        events = []
        t = 0.0
        while t < T_show:
            dt = np.random.exponential(1.0 / lam)
            t += dt
            if t <= T_show:
                events.append(t)

        ax.eventplot(events, lineoffsets=0, linelengths=0.6, color="#2d6cdf")
        ax.set_xlim(0, T_show)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Время")
        ax.set_yticks([])
        ax.set_title(f"Пример реализации потока, λ={lam}")
        ax.grid(True, alpha=0.2, axis="x")
        self.timeline_canvas.draw()


def main():
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
