import math
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from time import perf_counter

x, y = sym.symbols('x y')

# Simulation Parameters (altered by the user)
function = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2  # Function to be minimized
init_pop_x = np.array([-40, 40])    # Possible range of initial point's x value
init_pop_y = np.array([-40, 40])    # Possible range of initial point's y value
mutation_str = 10                   # Initial mutation standard variation (recommended much smaller than init_pop_x/y)
mutation_rate = 5                  # Mutation std dev change rate (recommended 5 (22% change) or [4,8])
rejection_correction = 5           # Number of times rejection change rate is lower than acceptance (recommended [4,7])
generations = 3000000               # Number of generation in simulation (recommended low numbers for plotting)
benchmarking = 1          # Number of times the simulation is repeated (use 1 to disable)

plot = True                # Whether to plot results or not (recommended False for larger number of generations)
plot_precision = 450        # Number of plot image points per axis
plot_levels = 100            # Number of plot colour levels

exit_condition = True       # Whether to stop simulation if close to "answer" (both x and y)
stop_x = np.array([0, 2])  # The range within which to stop the simulation (x)
stop_y = np.array([0, 2])  # The range within which to stop the simulation (y)

# Initialize Variables
timer = 0
gens = 0
total_gens = 0
fitness = sym.lambdify([x, y], function, "numpy")
sample_old = np.array([np.random.uniform(init_pop_x[0], init_pop_x[1]),
                       np.random.uniform(init_pop_y[0], init_pop_y[1])])
fitness_old = fitness(sample_old[0], sample_old[1])
init_point = sample_old

# Initialize Figure
if plot:
    plt.figure()
    x_val = np.linspace(init_pop_x[0], init_pop_x[1], plot_precision)
    y_val = np.linspace(init_pop_y[0], init_pop_y[1], plot_precision)
    x_val, y_val = np.meshgrid(x_val, y_val)
    z_val = fitness(x_val, y_val)
    levels = np.linspace(z_val.min(), z_val.max(), plot_levels)
    plt.contourf(x_val, y_val, z_val, levels=levels)
    plt.draw()
    plt.pause(0.1)

# Begin Evolution Algorithm loop
for j in range(benchmarking):
    gens = generations
    for i in range(generations):
        stopwatch = perf_counter()
        sample_new = sample_old + np.random.normal([0, 0], mutation_str)
        fitness_new = fitness(sample_new[0], sample_new[1])
        if fitness_new < fitness_old:
            sample_old = sample_new
            fitness_old = fitness_new
            mutation_str *= math.exp(1 / mutation_rate)         # increase std dev
        else:
            mutation_str *= math.exp(-1 / (rejection_correction * mutation_rate))  # decrease std dev
        timer += perf_counter() - stopwatch
        # update plot graph
        if plot:
            plt.plot(sample_old[0], sample_old[1], '+--', linewidth=2.0)
            plt.draw()
            plt.pause(0.001)
        # check exit condition
        if exit_condition:
            if stop_x[0] < sample_old[0] < stop_x[1] and stop_y[0] < sample_old[1] < stop_y[1]:
                gens = i
                break

    # Print results and statistics
    total_gens += gens
    print("Initial point [x,y] = " + str(init_point))
    print("Number of generations: " + str(total_gens))
    print("Run time (in seconds): " + str(timer))
    print("Final point [x,y] = " + str(sample_old))
    print("Final point fitness cost: " + str(fitness_old))
    sample_old = np.array([np.random.uniform(init_pop_x[0], init_pop_x[1]),
                           np.random.uniform(init_pop_y[0], init_pop_y[1])])
    fitness_old = fitness(sample_old[0], sample_old[1])
    init_point = sample_old

plt.show()



