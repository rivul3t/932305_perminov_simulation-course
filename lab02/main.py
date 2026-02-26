import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Simulation:

    def compute(self,
                T_left,
                T_right,
                T_0,
                X_left,
                X_right,
                time,
                rho,
                c,
                lam,
                h,
                tau):

        n = int((X_right - X_left) / h) + 1
        steps = int(time / tau)

        T = [T_0] * n
        T[0] = T_left
        T[n - 1] = T_right

        A = lam / h**2
        B = rho * c / tau + 2 * lam / h**2
        C = lam / h**2

        for _ in range(steps):

            alpha = [0.0] * n
            beta = [0.0] * n

            alpha[0] = 0.0
            beta[0] = T[0]

            for i in range(2, n - 1):
                Fi = - (rho * c / tau) * T[i]
                denom = B - C * alpha[i-1]
                alpha[i] = A / denom
                beta[i] = (C * beta[i-1] - Fi) / denom

            T_new = [0.0] * n
            T_new[-1] = T[-1]

            for i in range(n - 2, 0, -1):
                T_new[i] = alpha[i] * T_new[i + 1] + beta[i]

            T_new[0] = T[0]
            T = T_new

        return T



class App:

    def __init__(self, root):
        self.root = root
        self.root.title("Теплопроводность")

        self.sim = Simulation()

        self.frame = ttk.Frame(root)
        self.frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.create_input("T_left", "0")
        self.create_input("T_right", "0")
        self.create_input("T_initial", "100")
        self.create_input("X_left", "0")
        self.create_input("X_right", "1")
        self.create_input("Time", "2")
        self.create_input("rho", "7800")
        self.create_input("c", "500")
        self.create_input("Lambda", "50")
        self.create_input("h", "0.05")
        self.create_input("tau", "0.01")

        ttk.Button(self.frame, text="Запустить",
                   command=self.run).pack(pady=5)

        # График
        self.figure, self.ax = plt.subplots(figsize=(6, 5))
        self.ax.set_title("Распределение температуры")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("T")
        self.ax.grid()

        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT)

    def create_input(self, label, default):
        ttk.Label(self.frame, text=label).pack()
        entry = ttk.Entry(self.frame)
        entry.insert(0, default)
        entry.pack()
        setattr(self, label, entry)

    def run(self):

        T_left = float(self.T_left.get())
        T_right = float(self.T_right.get())
        T_0 = float(self.T_initial.get())
        X_left = float(self.X_left.get())
        X_right = float(self.X_right.get())
        time = float(self.Time.get())
        density = float(self.rho.get())
        specific_heat = float(self.c.get())
        conductivity = float(self.Lambda.get())
        h = float(self.h.get())
        tau = float(self.tau.get())

        T = self.sim.compute(
            T_left,
            T_right,
            T_0,
            X_left,
            X_right,
            time,
            density,
            specific_heat,
            conductivity,
            h,
            tau
        )

        n = len(T)
        x = np.linspace(X_left, X_right, n)

        self.ax.cla()
        self.ax.plot(x, T)
        self.ax.set_title("Распределение температуры")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("T")
        self.ax.grid()
        self.canvas.draw()

        center_temp = T[n // 2]
        print(f"Температура в центре через {time} c: {center_temp:.4f}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
