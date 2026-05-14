import numpy as np
import random
import math

class OptimizationEnv:
    def __init__(self, width, height, heat_sources, avoid_sources=None, valid_bounds=None):
        """
        heat_sources: list of dicts, e.g. [{"x": 100, "y": 200, "intensity": 500, "name": "CPU"}]
        avoid_sources: list of dicts for not recommended areas
        """
        self.width = width
        self.height = height
        self.heat_sources = heat_sources
        self.avoid_sources = avoid_sources if avoid_sources is not None else []
        if valid_bounds is None:
            self.valid_bounds = {'x_min': 0, 'x_max': width - 1, 'y_min': 0, 'y_max': height - 1}
        else:
            self.valid_bounds = valid_bounds

    def calculate_heat(self, x, y):
        # Calculate inverse-distance weighted heat at coordinate (x,y)
        total_heat = 0
        for src in self.heat_sources:
            dx = x - src["x"]
            dy = y - src["y"]
            dist = math.sqrt(dx**2 + dy**2)
            # intensity / (distance + 1) to avoid dividing by 0
            # Higher intensity means more heat radiated.
            total_heat += src["intensity"] / (dist + 1.0)
            
        # Add heat penalty for avoid areas (Tidak Disarankan)
        for avd in self.avoid_sources:
            dx = x - avd["x"]
            dy = y - avd["y"]
            dist = math.sqrt(dx**2 + dy**2)
            # Using explicit intensity or default penalty intensity
            intensity = avd.get("intensity", 5000)
            total_heat += intensity / (dist + 1.0)
            
        return total_heat

def simulated_annealing(env, initial_x, initial_y, initial_temp=100.0, cooling_rate=0.90, max_iter=50):
    current_x = initial_x
    current_y = initial_y
    current_heat = env.calculate_heat(current_x, current_y)
    
    best_x = current_x
    best_y = current_y
    best_heat = current_heat
    
    temp = initial_temp
    
    for i in range(max_iter):
        # Determine perturbation range based on temperature
        step = max(1, int(temp))
        next_x = current_x + random.randint(-step, step)
        next_y = current_y + random.randint(-step, step)
        
        # Clamp within valid bounds
        next_x = max(env.valid_bounds['x_min'], int(min(env.valid_bounds['x_max'], next_x)))
        next_y = max(env.valid_bounds['y_min'], int(min(env.valid_bounds['y_max'], next_y)))
        
        next_heat = env.calculate_heat(next_x, next_y)
        
        # Acceptance criteria
        delta = next_heat - current_heat
        if delta < 0:
            # Better solution, always accept
            current_x, current_y = next_x, next_y
            current_heat = next_heat
            if current_heat < best_heat:
                best_x, best_y = current_x, current_y
                best_heat = current_heat
        else:
            # Worse solution, accept probabilistically
            # Avoid overflow in exp
            try:
                prob = math.exp(-delta / temp)
            except OverflowError:
                prob = 0
                
            if random.random() < prob:
                current_x, current_y = next_x, next_y
                current_heat = next_heat
                
        # Cool down
        temp *= cooling_rate
        
    return best_x, best_y, best_heat

def hybrid_ga_sa(env, pop_size=50, generations=20, sa_iter=20, logs=None):
    if logs is None:
        logs = []
        
    global_best = {'heat': float('inf'), 'x': None, 'y': None, 'gen': -1}
    
    # Initialize Random Population
    population = [{'x': random.randint(env.valid_bounds['x_min'], env.valid_bounds['x_max']), 
                   'y': random.randint(env.valid_bounds['y_min'], env.valid_bounds['y_max'])} 
                  for _ in range(pop_size)]
    
    # Eval Fitness
    for ind in population:
        ind['heat'] = env.calculate_heat(ind['x'], ind['y'])
        
    for gen in range(generations):
        # Sort by heat (lower is better cooler is better)
        population.sort(key=lambda item: item['heat'])
        
        # Elitism: top 20% go to next gen directly
        next_gen = population[:int(pop_size * 0.2)]
        
        # Mating pool for crossover
        mating_pool = population[:int(pop_size * 0.5)]
        
        # Crossover & Mutation
        while len(next_gen) < pop_size:
            p1 = random.choice(mating_pool)
            p2 = random.choice(mating_pool)
            
            # Crossover: Average coordinates
            child_x = int((p1['x'] + p2['x']) / 2.0)
            child_y = int((p1['y'] + p2['y']) / 2.0)
            
            # Mutation (Random Coordinate Shift)
            if random.random() < 0.4: # 40% probability
                rx = random.randint(-30, 30)
                ry = random.randint(-30, 30)
                child_x = max(env.valid_bounds['x_min'], min(env.valid_bounds['x_max'], child_x + rx))
                child_y = max(env.valid_bounds['y_min'], min(env.valid_bounds['y_max'], child_y + ry))
            
            next_gen.append({'x': child_x, 'y': child_y, 'heat': env.calculate_heat(child_x, child_y)})
            
        population = next_gen
        
        # --- HYBRIDIZATION: Apply SA to the best 10 individuals to perform local search ---
        for i in range(min(10, len(population))):
            curr_ind = population[i]
            # Fast SA refinement
            ox, oy, oheat = simulated_annealing(env, curr_ind['x'], curr_ind['y'], initial_temp=50.0, cooling_rate=0.8, max_iter=sa_iter)
            population[i] = {'x': ox, 'y': oy, 'heat': oheat}
            
        # Sort once more after SA to check if improvements were made
        population.sort(key=lambda item: item['heat'])
        
        # Record current generation best
        current_best = population[0]
        logs.append(f"[Generasi {gen+1:02d}] Panas Terbaik: {current_best['heat']:.2f} di ({current_best['x']}, {current_best['y']})")
        
        # Track global best across all generations
        if current_best['heat'] < global_best['heat']:
            global_best = {
                'heat': current_best['heat'],
                'x': current_best['x'],
                'y': current_best['y'],
                'gen': gen + 1
            }
            
    # Add Summary to Logs
    logs.append("\n=== RINGKASAN ITERASI TERBAIK ===")
    logs.append(f"📌 Generasi Terbaik: Generasi {global_best['gen']}")
    logs.append(f"🎯 Koordinat Terbaik: X={global_best['x']}, Y={global_best['y']}")
    logs.append(f"🔥 Skor Panas Minimum: {global_best['heat']:.2f}")
    
    return global_best
