import tkinter as tk
import random
import math
import noise

EMPTY = 0
TREE = 1
BURNING = 2
BURNT = 3


class ForestFireApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Forest fire modeling")

        self.size = 100
        self.cell_size = 6

        self.display_mode = "fire"

        self.p_growth = tk.DoubleVar(value=0.01)
        self.p_lightning = tk.DoubleVar(value=0.0005)
        self.base_fire_prob = tk.DoubleVar(value=0.3)


        self.wind_strength = tk.DoubleVar(value=1.5)
        self.wind_direction = tk.DoubleVar(value=0)
        self.humidity = tk.DoubleVar(value=0.3)

        tk.Label(root, text="Вероятность роста дерева").pack()
        tk.Entry(root, textvariable=self.p_growth).pack()

        tk.Label(root, text="Базовая вероятность поджога дерева").pack()
        tk.Entry(root, textvariable=self.base_fire_prob).pack()

        tk.Label(root, text="Вероятность удара молнии").pack()
        tk.Entry(root, textvariable=self.p_lightning).pack()

        tk.Label(root, text="Сила ветра").pack()
        tk.Entry(root, textvariable=self.wind_strength).pack()

        tk.Label(root, text="Направление ветра(радианы)").pack()
        tk.Entry(root, textvariable=self.wind_direction).pack()

        tk.Label(root, text="Влажность (0-1)").pack()
        tk.Entry(root, textvariable=self.humidity).pack()

        tk.Button(root, text="Fire view", command=lambda: self.set_mode("fire")).pack()
        tk.Button(root, text="Terrain view", command=lambda: self.set_mode("terrain")).pack()

        tk.Button(root, text="Start", command=self.start).pack()
        tk.Button(root, text="Stop", command=self.stop).pack()

        self.canvas = tk.Canvas(
            root,
            width=self.size * self.cell_size,
            height=self.size * self.cell_size
        )
        self.canvas.pack()

        self.running = False

        self.grid = [[EMPTY for _ in range(self.size)]
                     for _ in range(self.size)]

        self.generate_heightmap()

        self.min_height = min(min(row) for row in self.height)
        self.max_height = max(max(row) for row in self.height)

    def start(self):
        self.running = True
        self.update()

    def stop(self):
        self.running = False

    def update(self):

        if not self.running:
            return

        new_grid = [[EMPTY for _ in range(self.size)]
                    for _ in range(self.size)]

        for i in range(self.size):
            for j in range(self.size):

                state = self.grid[i][j]

                if state == BURNING:
                    new_grid[i][j] = BURNT

                elif state == TREE:

                    p_fire = self.fire_probability(i, j)

                    if random.random() < p_fire:
                        new_grid[i][j] = BURNING

                    elif random.random() < self.p_lightning.get():
                        new_grid[i][j] = BURNING

                    else:
                        new_grid[i][j] = TREE

                elif state == EMPTY:

                    if random.random() < self.p_growth.get():
                        new_grid[i][j] = TREE

                elif state == BURNT:
                    new_grid[i][j] = EMPTY

        self.grid = new_grid
        self.draw()
        self.root.after(50, self.update)

    def fire_probability(self, x, y):
        base_prob = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:

                if dx == 0 and dy == 0:
                    continue

                nx = x + dx
                ny = y + dy

                if 0 <= nx < self.size and 0 <= ny < self.size:

                    if self.grid[nx][ny] == BURNING:

                        prob = self.base_fire_prob.get()
                        prob *= self.wind_factor(dx, dy)
                        prob *= self.humidity_factor()
                        prob *= self.slope_factor(x, y, nx, ny)
                        base_prob += prob


        return min(base_prob, 1)

    def wind_factor(self, dx, dy):
        wind_angle = self.wind_direction.get()
        wind_vec = (
            math.cos(wind_angle),
            math.sin(wind_angle)
        )
        dot = wind_vec[0]*dx + wind_vec[1]*dy

        return 1 + dot * self.wind_strength.get() * 0.3

    def humidity_factor(self):
        h = self.humidity.get()
        return max(0.1, 1 - h)

    def slope_factor(self, x, y, nx, ny):

        h_current = self.height[x][y]
        h_neighbor = self.height[nx][ny]

        delta_h = h_current - h_neighbor

        return math.exp(delta_h * 0.05)

    def draw(self):

        self.canvas.delete("all")

        for i in range(self.size):
            for j in range(self.size):

                if self.display_mode == "terrain":

                    color = self.height_color(i, j)

                else:

                    state = self.grid[i][j]

                    if state == EMPTY:
                        color = "white"
                    elif state == TREE:
                        color = "green"
                    elif state == BURNING:
                        color = "red"
                    else:
                        color = "black"

                self.canvas.create_rectangle(
                    j * self.cell_size,
                    i * self.cell_size,
                    (j + 1) * self.cell_size,
                    (i + 1) * self.cell_size,
                    fill=color,
                    outline=""
                )

    def generate_heightmap(self, scale=50.0, octaves=6, persistence=0.5, lacunarity=2.0):

        self.height = [[0 for _ in range(self.size)]
                       for _ in range(self.size)]

        for i in range(self.size):
            for j in range(self.size):

                # шум Перлина
                h = noise.pnoise2(
                    i / scale,
                    j / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=self.size,
                    repeaty=self.size,
                    base=2
                )

                self.height[i][j] = (h + 0.5)

    def height_color(self, x, y):

        h = self.height[x][y]

        norm = (h - self.min_height) / (self.max_height - self.min_height)
        value = int(255 * norm)
        return f'#{value:02x}{value:02x}{value:02x}'

    def set_mode(self, mode):
        self.display_mode = mode
        self.draw()

if __name__ == "__main__":

    root = tk.Tk()

    app = ForestFireApp(root)

    root.mainloop()