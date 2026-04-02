import tkinter as tk
from tkinter import ttk
import random
import math


def choice(options):
    epsilon = random.random()
    A = 1.
    k = 1
    while True:
        option, Pk = options[k-1]
        A = A - Pk
        if A <= epsilon:
            return option
        k += 1

# =============================
# YES/NO
# =============================

class YesNoTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="#121212")

        self.answers = [("YES", 0.5), ("NO", 0.5)]
        self.display = tk.Label(self, text="?", font=("Verdana", 64, "bold"),
                                bg="#121212", fg="#4cc9f0")

        self.question_entry = tk.Entry(self, font=("Verdana", 14), width=50, bg="#1f1f1f", fg="white")
        self.question_entry.pack(pady=(20, 10))

        self.display.pack(pady=80)

        self.btn = tk.Button(
            self,
            text="Ask",
            command=self.ask,
            font=("Verdana", 14, "bold"),
            bg="#1f1f1f",
            fg="white",
            activebackground="#4cc9f0",
            relief="flat",
            padx=25,
            pady=10
        )
        self.btn.pack()

    def ask(self):
        result = choice(self.answers)
        color = "#00e676" if result == "YES" else "#ff5252"
        self.display.config(text=result, fg=color)


# =============================
# MAGIC BALL
# =============================
class MagicBallTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="#121212")

        self.question_entry = tk.Entry(self, font=("Verdana", 14), width=50, bg="#1f1f1f", fg="white")
        self.question_entry.pack(pady=(20, 10))

        self.canvas = tk.Canvas(self, width=420, height=420, bg="#121212", highlightthickness=0)
        self.canvas.pack(pady=20)

        self.cx, self.cy = 210, 210
        self.radius = 150

        self.canvas.create_oval(
            self.cx - self.radius, self.cy - self.radius,
            self.cx + self.radius, self.cy + self.radius,
            fill="#1a1a2e", outline="#9d4edd", width=3
        )

        self.poly_radius = 90
        self.base_points = []
        for i in range(8):
            ang = math.radians(45 * i)
            x = self.cx + self.poly_radius * math.cos(ang)
            y = self.cy + self.poly_radius * math.sin(ang)
            self.base_points.append((x, y))

        self.polygon = self.canvas.create_polygon(
            [c for pt in self.base_points for c in pt],
            fill="#006d77", outline=""
        )

        self.text_obj = self.canvas.create_text(
            self.cx, self.cy,
            text="",
            fill="white",
            font=("Verdana", 14, "bold"),
            width=140
        )

        self.answers = [
            ("Definitely yes", 0.125),
            ("Most likely", 0.125),
            ("Looks good", 0.125),
            ("Ask later", 0.125),
            ("Unclear", 0.125),
            ("Probably not", 0.125),
            ("Very doubtful", 0.125),
            ("Definitely no", 0.125)
        ]

        self.angle = 0
        self.rotating = False

        self.canvas.bind("<Button-1>", self.on_click)

    def rotate(self, points, angle):
        rotated = []
        for x, y in points:
            dx = x - self.cx
            dy = y - self.cy
            rx = dx * math.cos(angle) - dy * math.sin(angle)
            ry = dx * math.sin(angle) + dy * math.cos(angle)
            rotated.append((rx + self.cx, ry + self.cy))
        return rotated

    def draw_polygon(self, points):
        flat = [c for pt in points for c in pt]
        self.canvas.coords(self.polygon, flat)

    def spin(self, step=0):
        if step < 50:
            self.angle += 0.25
            self.draw_polygon(self.rotate(self.base_points, self.angle))
            self.canvas.itemconfig(self.text_obj, text="")
            self.after(15, self.spin, step + 1)
        else:
            self.rotating = False
            self.angle = 0
            self.draw_polygon(self.base_points)
            self.show_result()

    def show_result(self):
        result = choice(self.answers)
        self.canvas.itemconfig(self.text_obj, text=result)

    def on_click(self, event):
        if self.rotating:
            return
        self.rotating = True
        self.spin()

class DecisionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Decision Helper")
        self.geometry("620x520")
        self.configure(bg="#121212")
        self.resizable(False, False)

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(expand=True, fill="both")

        self.yesno_tab = YesNoTab(self.tabs)
        self.tabs.add(self.yesno_tab, text="Yes / No")

        self.magic_tab = MagicBallTab(self.tabs)
        self.tabs.add(self.magic_tab, text="Magic Ball")

if __name__ == "__main__":
    app = DecisionApp()
    app.mainloop()