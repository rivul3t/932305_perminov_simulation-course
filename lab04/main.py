import random

class LCG:
    def __init__(self, a=1103515245, c=12345, m=2**31, seed=67):
        self.a = a
        self.c = c
        self.m = m
        self.state = seed % m

    def random(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

def generate_samples(generator, num_samples=100_000):
    return [generator.random() for _ in range(num_samples)]

def get_mean(samples):
    return sum(samples) / len(samples)

def get_variance(samples, mean: float):
    return sum((x - mean)**2 for x in samples) / len(samples)

def main():
    seed=83

    lcg = LCG(seed=seed)
    samples_lcg = generate_samples(lcg)
    mean_lcg = get_mean(samples_lcg)
    var_lcg = get_variance(samples_lcg, mean_lcg)

    random.seed(seed)
    samples_rnd = generate_samples(random)
    mean_rnd = get_mean(samples_rnd)
    var_rnd = get_variance(samples_rnd, mean_rnd)

    theoretical_mean = 0.5
    theoretical_var = 1. / 12

    print("LCG generator:")
    print(f"sample mean = {mean_lcg:.6f}, diff = {mean_lcg - theoretical_mean:.6e}")
    print(f"sample variance = {var_lcg:.6f}, theoretical variance = {theoretical_var:.6f}, diff = {var_lcg - theoretical_var:.6e}\n")

    print("python random generator:")
    print(f"sample mean = {mean_rnd:.6f}, diff = {mean_rnd - theoretical_mean:.6e}")
    print(f"sample variance = {var_rnd:.6f}, theoretical variance = {theoretical_var:.6f}, diff = {var_rnd - theoretical_var:.6e}")

if __name__ == '__main__':
    main()