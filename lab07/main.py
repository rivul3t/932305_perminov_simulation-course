import sys
import csv
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QGroupBox,
    QSlider, QFileDialog, QMessageBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

STATES = ["Ясно", "Облачно", "Пасмурно"]
STATE_COLORS_MPL = ["#FFD700", "#6aaef5", "#9499a3"]

DEFAULT_LAMBDA = np.array([
    [0.0, 0.3, 0.1],
    [0.4, 0.0, 0.3],
    [0.2, 0.2, 0.0],
])

DARK_BG   = "#12121e"
PANEL_BG  = "#1a1a2e"
CARD_BG   = "#16213e"
ACCENT    = "#2d6cdf"
TEXT      = "#cdd6f4"
MUTED     = "#6c7086"
BORDER    = "#313244"

OSC_BG    = "#0d0d1a"

class MarkovSim:
    """Continuous-time Markov chain simulation."""

    def __init__(self, lam: np.ndarray):
        self.n   = lam.shape[0]
        self.lam = lam.copy()

        self.rates = lam.sum(axis=1)

        self.P = np.zeros((self.n, self.n))
        for i in range(self.n):
            if self.rates[i] > 0:
                self.P[i] = lam[i] / self.rates[i]

        self.state = 0
        self.time  = 0.0
        self._schedule_next()

        self.history_times:  list[float] = [0.0]
        self.history_states: list[int]   = [0]

        self.time_in_state = np.zeros(self.n)

    def _schedule_next(self):
        r = self.rates[self.state]
        self._next_time = self.time + (np.random.exponential(1.0 / r) if r > 0 else 1e12)

    def advance(self, dt: float):
        end = self.time + dt
        while self._next_time <= end:
            elapsed = self._next_time - self.time
            self.time_in_state[self.state] += elapsed
            self.time = self._next_time

            self.state = np.random.choice(self.n, p=self.P[self.state])
            self._schedule_next()

            self.history_times.append(self.time)
            self.history_states.append(self.state)

        self.time_in_state[self.state] += end - self.time
        self.time = end

    @property
    def n_transitions(self) -> int:
        return len(self.history_times) - 1

    def empirical_dist(self) -> np.ndarray:
        total = self.time_in_state.sum()
        if total <= 0:
            return np.ones(self.n) / self.n
        return self.time_in_state / total

    @staticmethod
    def stationary_dist(lam: np.ndarray) -> np.ndarray:
        """Solve πQ = 0, Σπ = 1 for the generator matrix Q."""
        n = lam.shape[0]
        Q = lam.copy()
        for i in range(n):
            Q[i, i] = -lam[i].sum()

        A = Q.T.copy()
        A[-1, :] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0

        try:
            pi = np.linalg.solve(A, b)
            pi /= pi.sum()
        except np.linalg.LinAlgError:
            pi = np.ones(n) / n
        return pi


class StateIndicator(QWidget):
    """Three glowing circles representing Ясно / Облачно / Пасмурно."""

    _ACTIVE = [
        QColor(255, 215,   0),
        QColor(100, 149, 237),
        QColor(148, 153, 163),
    ]
    _INACTIVE = [
        QColor( 70,  55,   5),
        QColor( 25,  45,  90),
        QColor( 42,  43,  52),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = 0
        self.setMinimumHeight(110)

    def set_state(self, s: int):
        if s != self._state:
            self._state = s
            self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        r = min(w // 7, (h - 30) // 2)

        for i in range(3):
            cx = w * (2 * i + 1) // 6
            cy = h // 2 - 12

            if i == self._state:
                # outer glow
                glow = QColor(self._ACTIVE[i])
                glow.setAlpha(45)
                p.setBrush(QBrush(glow))
                p.setPen(Qt.PenStyle.NoPen)
                rg = int(r * 1.65)
                p.drawEllipse(cx - rg, cy - rg, 2 * rg, 2 * rg)

                p.setBrush(QBrush(self._ACTIVE[i]))
                pen = QPen(QColor(255, 255, 255, 160))
                pen.setWidth(2)
                p.setPen(pen)
            else:
                p.setBrush(QBrush(self._INACTIVE[i]))
                p.setPen(Qt.PenStyle.NoPen)

            p.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)

            colour = QColor(255, 255, 255) if i == self._state else QColor(120, 120, 130)
            p.setPen(QPen(colour))
            font = QFont()
            font.setPointSize(9)
            font.setBold(i == self._state)
            p.setFont(font)
            p.drawText(cx - r, cy + r + 6, 2 * r, 18,
                       Qt.AlignmentFlag.AlignCenter, STATES[i])


class MplCanvas(FigureCanvas):
    def __init__(self, figsize=(8, 3), bg=PANEL_BG):
        self.fig = Figure(figsize=figsize, tight_layout=True,
                          facecolor=bg)
        self.ax = self.fig.add_subplot(111, facecolor=bg)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)



class MainWindow(QMainWindow):
    TICK_MS      = 50
    PLOT_EVERY   = 4

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Марковская модель погоды")
        self.resize(1440, 820)

        self.sim: MarkovSim | None = None
        self.running    = False
        self.speed      = 5.0 
        self._tick_cnt  = 0

        self._timer = QTimer()
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        root = QHBoxLayout(root_widget)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        root.addWidget(self._build_left_panel(), 0)
        root.addWidget(self._build_right_panel(), 1)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(310)
        lay = QVBoxLayout(panel)
        lay.setSpacing(8)
        lay.setContentsMargins(0, 0, 0, 0)

        lay.addWidget(self._build_matrix_box())
        lay.addWidget(self._build_speed_box())
        lay.addWidget(self._build_control_box())
        lay.addWidget(self._build_stats_box())
        lay.addWidget(self._build_state_box())
        lay.addStretch()
        return panel

    def _build_matrix_box(self) -> QGroupBox:
        box = QGroupBox("Матрица интенсивностей")
        grid = QGridLayout(box)
        grid.setSpacing(5)

        for j, name in enumerate(STATES):
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"color:{MUTED}; font-size:11px;")
            grid.addWidget(lbl, 0, j + 1)

        self._lam_edits: list[list[QLineEdit]] = []
        for i in range(3):
            row_lbl = QLabel(STATES[i])
            row_lbl.setStyleSheet(f"color:{MUTED}; font-size:11px;")
            grid.addWidget(row_lbl, i + 1, 0)

            row: list[QLineEdit] = []
            for j in range(3):
                edit = QLineEdit()
                edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
                edit.setFixedWidth(60)
                if i == j:
                    edit.setText("—")
                    edit.setEnabled(False)
                    edit.setStyleSheet(
                        f"background:{DARK_BG}; color:{MUTED}; border:1px solid {BORDER};")
                else:
                    edit.setText(str(DEFAULT_LAMBDA[i, j]))
                grid.addWidget(edit, i + 1, j + 1)
                row.append(edit)
            self._lam_edits.append(row)

        return box

    def _build_speed_box(self) -> QGroupBox:
        box = QGroupBox("Скорость симуляции")
        lay = QVBoxLayout(box)

        row = QHBoxLayout()
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(1, 300)
        self._speed_slider.setValue(int(self.speed))
        self._speed_slider.valueChanged.connect(self._on_speed_changed)

        self._speed_lbl = QLabel(f"{int(self.speed)} дн/с")
        self._speed_lbl.setFixedWidth(70)
        self._speed_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._speed_lbl.setStyleSheet("font-weight:bold; color:#7eb8f7;")

        row.addWidget(self._speed_slider)
        row.addWidget(self._speed_lbl)
        lay.addLayout(row)
        return box

    def _build_control_box(self) -> QGroupBox:
        box = QGroupBox("Управление")
        lay = QVBoxLayout(box)

        row1 = QHBoxLayout()
        self._start_btn = QPushButton("▶  Старт")
        self._stop_btn  = QPushButton("⏸  Пауза")
        self._stop_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)
        row1.addWidget(self._start_btn)
        row1.addWidget(self._stop_btn)

        row2 = QHBoxLayout()
        self._reset_btn = QPushButton("↺  Сброс")
        self._save_btn  = QPushButton("💾  CSV")
        self._reset_btn.clicked.connect(self._on_reset)
        self._save_btn.clicked.connect(self._on_save)
        row2.addWidget(self._reset_btn)
        row2.addWidget(self._save_btn)

        lay.addLayout(row1)
        lay.addLayout(row2)
        return box

    def _build_stats_box(self) -> QGroupBox:
        box = QGroupBox("Статистика")
        lay = QVBoxLayout(box)
        lay.setSpacing(4)

        self._days_lbl = QLabel("Дней прошло: 0.0")
        self._days_lbl.setStyleSheet(
            "font-size:17px; font-weight:bold; color:#7eb8f7;")
        self._days_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._trans_lbl = QLabel("Переходов: 0")
        self._trans_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._trans_lbl.setStyleSheet(f"color:{MUTED};")

        lay.addWidget(self._days_lbl)
        lay.addWidget(self._trans_lbl)
        return box

    def _build_state_box(self) -> QGroupBox:
        box = QGroupBox("Текущее состояние")
        lay = QVBoxLayout(box)
        self._indicator = StateIndicator()
        lay.addWidget(self._indicator)
        return box

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setSpacing(10)
        lay.setContentsMargins(0, 0, 0, 0)

        osc_box = QGroupBox("История состояний")
        osc_lay = QVBoxLayout(osc_box)
        self._osc = MplCanvas(figsize=(10, 3), bg=OSC_BG)
        self._osc.fig.patch.set_facecolor(OSC_BG)
        self._osc.ax.set_facecolor(OSC_BG)
        osc_lay.addWidget(self._osc)
        lay.addWidget(osc_box, stretch=1)

        hist_box = QGroupBox("Распределение состояний")
        hist_lay = QVBoxLayout(hist_box)
        self._hist = MplCanvas(figsize=(10, 3))
        hist_lay.addWidget(self._hist)
        lay.addWidget(hist_box, stretch=1)

        return panel

    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {DARK_BG};
                color: {TEXT};
                font-size: 13px;
                font-family: "Segoe UI", "DejaVu Sans", sans-serif;
            }}
            QGroupBox {{
                border: 1px solid {BORDER};
                border-radius: 8px;
                margin-top: 12px;
                background: {CARD_BG};
                font-weight: bold;
                color: {TEXT};
                padding: 6px 4px 4px 4px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #89b4fa;
            }}
            QPushButton {{
                background: {ACCENT};
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover   {{ background: #3a7ef5; }}
            QPushButton:pressed {{ background: #1f4b9c; }}
            QPushButton:disabled {{ background: {BORDER}; color: {MUTED}; }}
            QLineEdit {{
                padding: 5px 7px;
                border: 1px solid {BORDER};
                border-radius: 5px;
                background: #1e1e2e;
                color: {TEXT};
            }}
            QLineEdit:focus {{ border-color: {ACCENT}; }}
            QSlider::groove:horizontal {{
                border: none;
                height: 6px;
                background: {BORDER};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT};
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT};
                border-radius: 3px;
            }}
            QLabel {{ color: {TEXT}; }}
        """)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_speed_changed(self, v: int):
        self.speed = float(v)
        self._speed_lbl.setText(f"{v} дн/с")

    def _read_lambda(self) -> np.ndarray | None:
        lam = np.zeros((3, 3))
        try:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        txt = self._lam_edits[i][j].text().replace(",", ".")
                        v = float(txt)
                        if v < 0:
                            raise ValueError
                        lam[i, j] = v
        except ValueError:
            QMessageBox.critical(self, "Ошибка",
                                 "Интенсивности должны быть неотрицательными числами.")
            return None

        if lam.sum() == 0:
            QMessageBox.critical(self, "Ошибка",
                                 "Все интенсивности нулевые — переходов не будет.")
            return None
        return lam

    def _on_start(self):
        if self.sim is None:
            lam = self._read_lambda()
            if lam is None:
                return
            self.sim = MarkovSim(lam)
            self._redraw_all()

        self.running = True
        self._timer.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

    def _on_stop(self):
        self.running = False
        self._timer.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _on_reset(self):
        self._on_stop()
        self.sim = None
        self._tick_cnt = 0
        self._indicator.set_state(0)
        self._days_lbl.setText("Дней прошло: 0.0")
        self._trans_lbl.setText("Переходов: 0")
        self._clear_osc()
        self._clear_hist()

    def _on_save(self):
        if self.sim is None:
            QMessageBox.warning(self, "Нет данных", "Сначала запустите симуляцию.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить CSV", "weather_history.csv", "CSV (*.csv)")
        if not path:
            return

        emp  = self.sim.empirical_dist()
        theo = MarkovSim.stationary_dist(self.sim.lam)

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["# Матрица интенсивностей"])
            header = ["", ] + STATES
            w.writerow(header)
            for i in range(3):
                row_vals = [STATES[i]]
                for j in range(3):
                    row_vals.append(
                        "—" if i == j else str(self.sim.lam[i, j]))
                w.writerow(row_vals)

            w.writerow([])
            w.writerow(["# Стационарное распределение"])
            w.writerow(["Состояние", "Теоретическое", "Эмпирическое"])
            for i in range(3):
                w.writerow([STATES[i], f"{theo[i]:.6f}", f"{emp[i]:.6f}"])

            w.writerow([])
            w.writerow(["# История переходов"])
            w.writerow(["День", "Состояние (1=Ясно, 2=Облачно, 3=Пасмурно)"])
            for t, s in zip(self.sim.history_times, self.sim.history_states):
                w.writerow([f"{t:.6f}", s + 1])

        QMessageBox.information(self, "Сохранено", f"Файл сохранён:\n{path}")

    def _tick(self):
        if self.sim is None:
            return

        dt = self.speed * self.TICK_MS / 1000.0
        self.sim.advance(dt)

        self._indicator.set_state(self.sim.state)
        self._days_lbl.setText(f"Дней прошло: {self.sim.time:.1f}")
        self._trans_lbl.setText(f"Переходов: {self.sim.n_transitions}")

        self._tick_cnt += 1
        if self._tick_cnt % self.PLOT_EVERY == 0:
            self._redraw_all()


    def _redraw_all(self):
        self._draw_osc()
        self._draw_hist()

    def _clear_osc(self):
        ax = self._osc.ax
        ax.clear()
        ax.set_facecolor(OSC_BG)
        self._osc.fig.patch.set_facecolor(OSC_BG)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(STATES, color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(True, color="#1e1e3a", linestyle="--", linewidth=0.6)
        self._osc.draw()

    def _draw_osc(self):
        if self.sim is None:
            return

        ax = self._osc.ax
        ax.clear()
        ax.set_facecolor(OSC_BG)
        self._osc.fig.patch.set_facecolor(OSC_BG)

        times  = self.sim.history_times
        states = self.sim.history_states
        t_end  = self.sim.time

        # rolling window
        window  = max(80.0, t_end * 0.15)
        t_start = max(0.0, t_end - window)

        # find the last transition at or before t_start
        ht = np.asarray(times)
        idx = int(np.searchsorted(ht, t_start, side="right")) - 1
        idx = max(0, idx)

        seg_t = times[idx:]
        seg_s = states[idx:]

        # draw horizontal segments coloured by state
        for k in range(len(seg_t) - 1):
            x0 = max(seg_t[k],   t_start)
            x1 = min(seg_t[k+1], t_end)
            if x1 <= x0:
                continue
            s = seg_s[k]
            ax.plot([x0, x1], [s + 1, s + 1],
                    color=STATE_COLORS_MPL[s], linewidth=2.8,
                    solid_capstyle="butt")
            # vertical connector
            if k + 1 < len(seg_t) - 1:
                ns = seg_s[k + 1]
                ax.plot([x1, x1], [s + 1, ns + 1],
                        color="#555577", linewidth=1.2, linestyle=":")

        # last ongoing segment
        x0 = max(seg_t[-1], t_start)
        s  = seg_s[-1]
        ax.plot([x0, t_end], [s + 1, s + 1],
                color=STATE_COLORS_MPL[s], linewidth=2.8,
                solid_capstyle="butt")

        # legend markers (coloured dots)
        for i, (name, col) in enumerate(zip(STATES, STATE_COLORS_MPL)):
            ax.plot([], [], "o", color=col, label=name, markersize=6)

        ax.set_xlim(t_start, max(t_end + 0.5, t_start + 1))
        ax.set_ylim(0.5, 3.5)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(STATES, color=TEXT, fontsize=9)
        ax.set_xlabel("День", color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_color(BORDER)
        ax.grid(True, axis="x", color="#1e1e3a",
                linestyle="--", linewidth=0.6, alpha=0.8)

        leg = ax.legend(loc="upper left", fontsize=8,
                        framealpha=0.3, labelcolor="white",
                        facecolor="#1a1a2e", edgecolor=BORDER)
        self._osc.draw()

    def _clear_hist(self):
        ax = self._hist.ax
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        self._hist.fig.patch.set_facecolor(PANEL_BG)
        self._hist.draw()

    def _draw_hist(self):
        if self.sim is None:
            return

        ax = self._hist.ax
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        self._hist.fig.patch.set_facecolor(PANEL_BG)

        emp  = self.sim.empirical_dist()
        theo = MarkovSim.stationary_dist(self.sim.lam)

        x     = np.arange(3)
        width = 0.32

        bars_e = ax.bar(x - width / 2, emp,  width,
                        label="Эмпирическое",   color="#4a9eff",
                        alpha=0.85, edgecolor="#2d6cdf", linewidth=1.2)
        bars_t = ax.bar(x + width / 2, theo, width,
                        label="Теоретическое",  color="#ff7f50",
                        alpha=0.85, edgecolor="#cc5533", linewidth=1.2)

        for bar in (*bars_e, *bars_t):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.012, f"{h:.3f}",
                    ha="center", va="bottom",
                    fontsize=8, color=TEXT)

        ax.set_xticks(x)
        ax.set_xticklabels(STATES, fontsize=10, color=TEXT)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Доля времени", color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_color(BORDER)
        ax.grid(True, axis="y", color=BORDER, linewidth=0.6, alpha=0.7)

        leg = ax.legend(fontsize=9, framealpha=0.4, labelcolor=TEXT,
                        facecolor=PANEL_BG, edgecolor=BORDER)

        self._hist.fig.tight_layout()
        self._hist.draw()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
