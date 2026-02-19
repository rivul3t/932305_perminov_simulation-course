import tkinter as tk
from tkinter import ttk
import numpy as np
from math import sqrt
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Моделирование полёта тела")

        self.frame = ttk.Frame(root)
        self.frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.create_input("Начальная скорость (м/с)", "50")
        self.create_input("Угол (градусы)", "45")
        self.create_input("Масса (кг)", "1")
        self.create_input("Радиус (м)", "0.05")
        self.create_input("Плотность воздуха (кг/м³)", "1.29")
        self.create_input("Коэффициент сопротивления Cd", "0.15")
        self.create_input("Шаг dt", "0.05")
        self.create_input("Ускорение свободного падения (м/с²)", "9.81")

        ttk.Button(self.frame, text="Запустить",
                   command=self.run_simulation).pack(pady=5)

        ttk.Button(self.frame, text="Очистить график",
                   command=self.clear_plot).pack(pady=5)

        self.figure, self.ax = plt.subplots(figsize=(6, 5))
        self.ax.set_title("Траектория полёта")
        self.ax.set_xlabel("X (м)")
        self.ax.set_ylabel("Y (м)")
        self.ax.grid()

        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT)

    def create_input(self, label, default):
        ttk.Label(self.frame, text=label).pack()
        entry = ttk.Entry(self.frame)
        entry.insert(0, default)
        entry.pack()
        setattr(self, label, entry)

    def _step(self, state, dt, k, g):
        v = sqrt(state['v_x']**2 + state['v_y']**2)
        state['v_x'] = state['v_x'] - k * state['v_x'] * v * dt
        state['v_y'] = state['v_y'] - (g + k * state['v_y'] * v) * dt
        state['x'] = state['x'] + state['v_x'] * dt
        state['y'] = state['y'] + state['v_y'] * dt

        return state


    def run_simulation(self):
        v0 = float(getattr(self, "Начальная скорость (м/с)").get())
        angle = np.radians(float(getattr(self, "Угол (градусы)").get()))
        m = float(getattr(self, "Масса (кг)").get())
        r = float(getattr(self, "Радиус (м)").get())
        rho = float(getattr(self, "Плотность воздуха (кг/м³)").get())
        Cd = float(getattr(self, "Коэффициент сопротивления Cd").get())
        dt = float(getattr(self, "Шаг dt").get())
        g = float(getattr(self, "Ускорение свободного падения (м/с²)").get())

        k = Cd * (np.pi * r**2) * rho / (2 * m)

        vx0 = v0 * np.cos(angle)
        vy0 = v0 * np.sin(angle)

        state = {
            "x": 0,
            "y": 0,
            "v_x": vx0,
            "v_y": vy0
        }

        x_vals = []
        y_vals = []

        max_height = 0
        time = 0

        while state["y"] >= 0:
            x_vals.append(state["x"])
            y_vals.append(state["y"])
            max_height = max(max_height, state["y"])

            state = self._step(state, dt, k, g)
            time += dt

        distance = state["x"]
        v_final = sqrt(state["v_x"]**2 + state["v_y"]**2)

        self.ax.plot(x_vals, y_vals, label=f"dt={dt}")
        self.ax.legend()
        self.canvas.draw()

        print(f"Дальность: {distance:.2f} м")
        print(f"Максимальная высота: {max_height:.2f} м")
        print(f"Скорость в конечной точке: {v_final:.2f} м/c")
        print(f"Время полёта: {time:.2f} с")
        print("-"*30)

    def clear_plot(self):
        self.ax.cla()
        self.ax.set_title("Траектория полёта")
        self.ax.set_xlabel("X (м)")
        self.ax.set_ylabel("Y (м)")
        self.ax.grid()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationApp(root)
    root.mainloop()
