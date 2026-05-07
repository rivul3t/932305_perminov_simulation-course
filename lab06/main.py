import sys
import numpy as np
from scipy.stats import chisquare, chi2, norm
import random

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QGroupBox,
    QHeaderView,
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


VALUES = np.array([1, 2, 3, 4, 5], dtype=int)
DEFAULT_PROBS = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float)
NS = [10, 100, 1000, 10000]

def generate_discrete(n: int, probs: np.ndarray) -> np.ndarray:
    return np.random.choice(VALUES, size=n, p=probs)



def analyze_discrete(sample: np.ndarray, n: int, probs: np.ndarray):
    true_mean = np.sum(VALUES * probs)
    true_var = np.sum((VALUES - true_mean) ** 2 * probs)

    unique, counts = np.unique(sample, return_counts=True)
    freq_map = dict(zip(unique, counts))

    counts = np.array([np.sum(sample == v) for v in VALUES])
    emp_p = counts / n

    mean = np.mean(sample)
    var = np.var(sample)

    rel_mean_err = abs(mean - true_mean) / abs(true_mean) if true_mean != 0 else 0.0
    rel_var_err = abs(var - true_var) / abs(true_var) if true_var != 0 else 0.0

    chi_val = 0.0

    for i, v in enumerate(VALUES):
        expected = n * probs[i]
        observed = freq_map.get(v, 0)

        chi_val += ((observed - expected) ** 2) / expected

    df = len(VALUES) - 1

    chi_crit = chi2.ppf(0.95, df)

    p_value = 1 - chi2.cdf(chi_val, df)

    return emp_p, mean, var, rel_mean_err, rel_var_err, chi_val, chi_crit, p_value

class MplCanvas(FigureCanvas):
    def __init__(self, nrows=1, ncols=1, figsize=(8, 6)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.axes = self.fig.subplots(nrows, ncols)
        super().__init__(self.fig)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распределяшки — PyQt")
        self.resize(1200, 820)

        self.probs = DEFAULT_PROBS.copy()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.discrete_tab = QWidget()
        self.normal_tab = QWidget()

        self.tabs.addTab(self.discrete_tab, "Дискретная СВ")
        self.tabs.addTab(self.normal_tab, "Нормальная СВ")

        self.build_discrete_tab()
        self.build_normal_tab()

        self.apply_style()

    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background: #f5f7fb;
            }
            QWidget {
                font-size: 13px;
            }
            QTabWidget::pane {
                border: 1px solid #cfd8e3;
                background: white;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #e9eef5;
                padding: 10px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: white;
                font-weight: bold;
            }
            QGroupBox {
                border: 1px solid #d7dee8;
                border-radius: 8px;
                margin-top: 10px;
                background: white;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QPushButton {
                background: #2d6cdf;
                color: white;
                border: none;
                padding: 8px 14px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: #2458b8;
            }
            QPushButton:pressed {
                background: #1f4b9c;
            }
            QLineEdit {
                padding: 6px 8px;
                border: 1px solid #cfd8e3;
                border-radius: 6px;
                background: white;
            }
            QTableWidget {
                background: white;
                border: 1px solid #cfd8e3;
                border-radius: 8px;
                gridline-color: #e0e6ef;
            }
            QHeaderView::section {
                background: #eef3f9;
                padding: 6px;
                border: none;
                border-bottom: 1px solid #d7dee8;
                font-weight: bold;
            }
        """)

    def build_discrete_tab(self):
        layout = QVBoxLayout(self.discrete_tab)

        controls_box = QGroupBox("Параметры распределения")
        controls_layout = QHBoxLayout(controls_box)

        probs_box = QWidget()
        probs_layout = QGridLayout(probs_box)
        probs_layout.setHorizontalSpacing(10)
        probs_layout.setVerticalSpacing(8)

        self.prob_inputs = []
        for i, v in enumerate(VALUES):
            label = QLabel(f"P({v})")
            edit = QLineEdit(str(DEFAULT_PROBS[i]))
            edit.setFixedWidth(80)
            edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.prob_inputs.append(edit)

            probs_layout.addWidget(label, 0, i)
            probs_layout.addWidget(edit, 1, i)

        controls_layout.addWidget(probs_box)

        btn_box = QWidget()
        btn_layout = QVBoxLayout(btn_box)
        btn_layout.setContentsMargins(20, 0, 0, 0)

        self.uniform_btn = QPushButton("Равномерное")
        self.uniform_btn.clicked.connect(self.set_uniform_probs)

        self.run_discrete_btn = QPushButton("Запустить моделирование")
        self.run_discrete_btn.clicked.connect(self.run_discrete)

        btn_layout.addWidget(self.uniform_btn)
        btn_layout.addWidget(self.run_discrete_btn)
        btn_layout.addStretch()

        controls_layout.addWidget(btn_box)
        controls_layout.addStretch()

        layout.addWidget(controls_box)

        table_box = QGroupBox("Результаты")
        table_layout = QVBoxLayout(table_box)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "N",
            "Среднее",
            "Дисперсия",
            "Отн. погр. ср.",
            "Отн. погр. дисперсии",
            "χ²",
            "χ² критическое",
            "p-value",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)

        table_layout.addWidget(self.table)
        layout.addWidget(table_box)

        plots_box = QGroupBox("Гистограммы")
        plots_layout = QVBoxLayout(plots_box)

        self.discrete_canvas = MplCanvas(nrows=2, ncols=2, figsize=(10, 7))
        plots_layout.addWidget(self.discrete_canvas)

        layout.addWidget(plots_box, stretch=1)

    def set_uniform_probs(self):
        for edit in self.prob_inputs:
            edit.setText("0.2")
        self.probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float)

    def read_probs(self):
        try:
            probs = np.array([float(edit.text().replace(",", ".")) for edit in self.prob_inputs], dtype=float)

            if np.any(probs < 0):
                raise ValueError("Вероятности не могут быть отрицательными.")

            s = probs.sum()
            if not np.isclose(s, 1.0, atol=1e-6):
                raise ValueError("Сумма вероятностей должна быть равна 1.")

            return probs
        except Exception:
            QMessageBox.critical(
                self,
                "Ошибка",
                "Невалидные вероятности.\n"
            )
            return None

    def run_discrete(self):
        probs = self.read_probs()
        if probs is None:
            return

        self.probs = probs.copy()

        self.table.setRowCount(0)

        axes = self.discrete_canvas.axes.ravel()
        for ax in axes:
            ax.clear()

        bin_edges = np.arange(VALUES.min() - 0.5, VALUES.max() + 1.5, 1)

        for i, n in enumerate(NS):
            sample = generate_discrete(n, self.probs)
            emp_p, mean, var, rm, rv, chi_sq, chi_crit, p_value = analyze_discrete(sample, n, self.probs)

            row = self.table.rowCount()
            self.table.insertRow(row)

            values = [
                str(n),
                f"{mean:.4f}",
                f"{var:.4f}",
                f"{rm:.4f}",
                f"{rv:.4f}",
                f"{chi_sq:.4f}",
                f"{chi_crit:.4f}",                
                f"{p_value:.4f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)

            ax = axes[i]
            ax.hist(sample, bins=bin_edges, density=True, alpha=0.85, edgecolor="black")

            ax.set_xticks(VALUES)
            ax.set_ylim(0, 1)
            ax.set_title(f"N = {n}")
            ax.grid(True, alpha=0.25)

        self.discrete_canvas.fig.suptitle("Дискретное распределение", fontsize=14)
        self.discrete_canvas.draw()

    def build_normal_tab(self):
        layout = QVBoxLayout(self.normal_tab)

        controls_box = QGroupBox("Управление")
        controls_layout = QHBoxLayout(controls_box)

        self.bins_input = QLineEdit("12")
        self.bins_input.setFixedWidth(80)
        self.bins_input.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.mean = QLineEdit("0")
        self.mean.setFixedWidth(80)
        self.mean.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.sigma = QLineEdit("2")
        self.sigma.setFixedWidth(80)
        self.sigma.setAlignment(Qt.AlignmentFlag.AlignCenter)

        controls_layout.addWidget(QLabel("Количество бинов:"))
        controls_layout.addWidget(self.bins_input)

        controls_layout.addWidget(QLabel("Среднее:"))
        controls_layout.addWidget(self.mean)

        controls_layout.addWidget(QLabel("Среднеквадратичное отклонение:"))
        controls_layout.addWidget(self.sigma)

        self.run_normal_btn = QPushButton("Запустить моделирование")
        self.run_normal_btn.clicked.connect(self.run_normal)

        controls_layout.addWidget(self.run_normal_btn)
        controls_layout.addStretch()

        self.normal_table = QTableWidget(0, 8)
        self.normal_table.setHorizontalHeaderLabels([
            "N",
            "Среднее",
            "Дисперсия",
            "Отн. погр. ср.",
            "Отн. погр. дисперсии",
            "χ²",
            "χ² критическое",
            "p-value",
        ])
        self.normal_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.normal_table.verticalHeader().setVisible(False)
        
        layout.addWidget(self.normal_table)

        controls_layout.addWidget(self.run_normal_btn)
        controls_layout.addStretch()

        layout.addWidget(controls_box)

        plots_box = QGroupBox("Гистограмма и теоретическая плотность")
        plots_layout = QVBoxLayout(plots_box)

        self.normal_canvas = MplCanvas(nrows=2, ncols=2, figsize=(10, 7))
        plots_layout.addWidget(self.normal_canvas)

        layout.addWidget(plots_box, stretch=1)

    def run_normal(self):
        try:
            bins = int(self.bins_input.text())
            if bins < 2:
                raise ValueError
        except:
            QMessageBox.critical(self, "Ошибка", "Введите число бинов >1")
            return

        axes = self.normal_canvas.axes.ravel()

        for ax in axes:
            ax.clear()

        self.normal_table.setRowCount(0)

        mean, sigma = float(self.mean.text()), float(self.sigma.text())

        x = np.linspace(mean - sigma * 4, mean + sigma * 4, 800)
        pdf = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x-mean)/sigma)**2 / 2)

        for i, n in enumerate(NS):
            sample = self.generate_normal(n)

            mean, var, rm, rv, chi2_val, p_value, chi_crit = self.analyze_normal(sample, n, bins)

            row = self.normal_table.rowCount()
            self.normal_table.insertRow(row)

            vals = [
                str(n),
                f"{mean:.4f}",
                f"{var:.4f}",
                f"{rm:.4f}",
                f"{rv:.4f}",
                f"{chi2_val:.4f}",
                f"{chi_crit:.4f}",
                f"{p_value:.4f}",
            ]

            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.normal_table.setItem(row, col, item)

            ax = axes[i]
            ax.hist(sample, bins=bins, density=True, alpha=0.7)

            ax.plot(x, pdf, linewidth=2)
            ax.set_title(f"N = {n}")
            ax.grid(True, alpha=0.25)

        self.normal_canvas.fig.suptitle("Нормальное распределение", fontsize=14)
        self.normal_canvas.draw()

    def generate_normal(self, n: int) -> np.ndarray:
        mean, sigma = float(self.mean.text()), float(self.sigma.text())

        u1 = np.random.rand(n // 2)
        u2 = np.random.rand(n // 2)


        z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

        z0 = mean + sigma * z0
        z1 = mean + sigma * z1

        res = np.empty(n)
        res[0::2] = z0
        res[1::2] = z1

        return res

    def analyze_normal(self, sample: np.ndarray, n: int, bins: int = 6):
    
        mean, sigma = float(self.mean.text()), float(self.sigma.text())
    
        true_mean = mean
        true_var = sigma ** 2
    
        mean = np.mean(sample)
        var = np.var(sample)
    
        rel_mean_err = abs(mean - true_mean)
        rel_var_err = abs(var - true_var)
    
        counts, bin_edges = np.histogram(sample, bins=bins)
    
        O = counts
    
        P = []
        for i in range(len(bin_edges) - 1):
            p = norm.cdf(bin_edges[i+1], loc=mean, scale=sigma) - norm.cdf(bin_edges[i], loc=mean, scale=sigma)
            P.append(p)
    
        P = np.array(P)
        E = n * P
    
        mask = E > 0
        O = O[mask]
        E = E[mask]
    
        chi_val = np.sum((O - E) ** 2 / E)
    
        df = len(O) - 1
    
        chi_crit = chi2.ppf(0.95, df)
        p_value = 1 - chi2.cdf(chi_val, df)
    
        return mean, var, rel_mean_err, rel_var_err, chi_val, p_value, chi_crit

def main():
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()