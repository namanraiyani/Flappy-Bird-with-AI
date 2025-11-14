import sys
import json
import numpy as np
import pygame
import os

# Screen Constants
DISPLAY_W = 600
DISPLAY_H = 800
FLOOR_LEVEL = 730
FRAME_RATE = 30
SPAWN_X = 700
TUBE_DISTANCE = 300
FONT_STYLE = "comicsans"

pygame.init()
display_surface = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
pygame.display.set_caption("NeuroEvolution Simulation")
stats_font = pygame.font.SysFont(FONT_STYLE, 24)

from game_objects.bird import FlapBird
from game_objects.obstacle import Pipe
from game_objects.ground import ScrollingFloor
from game_objects.agent import Agent

background_img = pygame.transform.scale(pygame.image.load(os.path.join("game_objects/assets", "background.png")).convert_alpha(), (600, 900))

MAX_RENDER_LIMIT = 20

class GeneticSettings:
    def __init__(self, inputs, hidden_layers, outputs, population_count, top_performers_rate, mutation_chance, weight_volatility, bias_volatility, total_epochs, rng_seed):
        self.inputs = inputs
        self.hidden_layers = hidden_layers
        self.outputs = outputs
        self.population_count = population_count
        self.top_performers_rate = top_performers_rate
        self.mutation_chance = mutation_chance
        self.weight_volatility = weight_volatility
        self.bias_volatility = bias_volatility
        self.total_epochs = total_epochs
        self.rng_seed = rng_seed
        
def breed_next_gen(current_pop, fitness_scores, settings, rng: np.random.Generator):
    ranked_indices = np.argsort(fitness_scores)[::-1]
    
    keep_count = max(1, int(len(current_pop) * settings.top_performers_rate))
    survivors = [current_pop[i].create_replica() for i in ranked_indices[:keep_count]]
    
    champion = survivors[0]
    champion_score = float(fitness_scores[ranked_indices[0]])
    
    next_pop = []
    next_pop.extend(survivors)
    
    while len(next_pop) < len(current_pop):
        random_parent = survivors[int(rng.integers(0, keep_count))]
        offspring = random_parent.create_replica()
        offspring.apply_genetic_mutation(
            weight_variance = settings.weight_volatility,   
            bias_variance = settings.bias_volatility,
            mutation_rate = settings.mutation_chance
        )
        next_pop.append(offspring)
        
    return next_pop, champion, champion_score

def export_brain_data(brain, filename):
    packet = {
        "layer_structure": brain.layer_structure,
        "synapses": [w.tolist() for w in brain.synaptic_weights],
        "biases": [b.tolist() for b in brain.neuron_biases]
    }
    with open(filename, "w") as file_out:
        json.dump(packet, file_out)

def get_nearest_obstacle(avatar_x, obstacles):
    if len(obstacles) == 1:
        return obstacles[0]
    first_obs = obstacles[0]
    if avatar_x > first_obs.pos_x + first_obs.TUBE_INVERTED.get_width():
        return obstacles[1]
    return first_obs

def compute_inputs(avatar, obstacle):
    gap_center_y = (obstacle.opening_height + obstacle.bottom_y) / 2.0
    gap_radius = (obstacle.bottom_y - obstacle.opening_height) / 2.0
    avatar_center_y = avatar.pos_y + avatar.current_sprite.get_height() / 2.0
    
    delta_x = (obstacle.pos_x - avatar.pos_x)
    delta_y = (gap_center_y - avatar_center_y)
    velocity_y = avatar.vertical_velocity
    
    return np.array([delta_x, delta_y, velocity_y, gap_radius], dtype=np.float32)

def run_simulation_cycle(window, timer, text_renderer, brains, epoch_idx):
    floor = ScrollingFloor(FLOOR_LEVEL)
    obstacles = [Pipe(SPAWN_X)]
    
    entities = []
    for brain in brains:
        entities.append({
            "controller": brain,     
            "avatar": FlapBird(230, 350),
            "is_active": True,
            "performance": 0.0,
            "tubes_cleared": set()
        })
    
    is_simulating = True
    while is_simulating:
        timer.tick(FRAME_RATE)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
                
        tubes_to_remove = []
        for tube in obstacles:
            tube.update_position()     
            if tube.pos_x + tube.TUBE_INVERTED.get_width() < 0:
                tubes_to_remove.append(tube)
        
        if obstacles[-1].pos_x < DISPLAY_W - TUBE_DISTANCE:
            obstacles.append(Pipe(DISPLAY_W))
        
        for t in tubes_to_remove:
            obstacles.remove(t)
            
        floor.update_position()
        
        active_count = 0
        for entity in entities:
            if not entity["is_active"]:
                continue
            
            hero = entity["avatar"]
            mind = entity["controller"]
            
            target_tube = get_nearest_obstacle(avatar_x=hero.pos_x, obstacles=obstacles)
            sensor_data = compute_inputs(avatar=hero, obstacle=target_tube)  
            
            decision = mind.decide_action(sensor_data)
            if decision == 1:
                hero.perform_flap()
            
            hero.update_physics()
            
            if (hero.pos_y + hero.current_sprite.get_height() - 10 >= FLOOR_LEVEL) or (hero.pos_y < -50):
                entity["is_active"]= False
                continue
            
            has_crashed = False
            for tube in obstacles:
                if tube.check_crash(avatar=hero):
                    entity["is_active"] = False
                    has_crashed = True
                    break
                
            if has_crashed:
                continue
            
            entity["performance"] += 0.05
            
            for i, tube in enumerate(obstacles):
                tube_right_edge = tube.pos_x + tube.TUBE_INVERTED.get_width()
                if (tube_right_edge < hero.pos_x) and (i not in entity["tubes_cleared"]):
                    entity["tubes_cleared"].add(i)
                    entity["performance"] += 1.0
                    
            active_count += 1
        
        if active_count == 0:
            is_simulating = False
        
        window.blit(background_img, (0, 0))
        for tube in obstacles:
            tube.render_tubes(surface=window)
        
        rendered_count = 0
        for entity in entities:
            if not entity["is_active"]:
                continue
            if rendered_count >= MAX_RENDER_LIMIT:
                break
            
            entity["avatar"].render_avatar(surface=window)
            rendered_count += 1
        
        floor.render_to_surface(surface=window)
        
        max_fitness = max(e["performance"] for e in entities) if entities else 0.0
        stats_text = [
            f"Epoch: {epoch_idx}",
            f"Survivors: {active_count}/{len(entities)}",
            f"High Score: {max_fitness:.1f}"
        ]
        
        text_y = 8
        for line in stats_text:
            txt_surface = text_renderer.render(line, True, (0, 0, 0)) 
            window.blit(txt_surface, (10, text_y))
            text_y += 26

        pygame.display.flip()
    
    return np.array([e["performance"] for e in entities], dtype=np.float32)

def start_evolution_process():
    game_clock = pygame.time.Clock()
    
    config = GeneticSettings(inputs=4,
                    hidden_layers=(32,32),
                    outputs=1,
                    population_count=50,
                    top_performers_rate=0.10,
                    mutation_chance=0.12,
                    weight_volatility=0.12,
                    bias_volatility=0.06,
                    total_epochs=float('inf'),
                    rng_seed=42)
    
    rng_engine = np.random.default_rng(config.rng_seed)
    
    population = []
    for _ in range(config.population_count):        
        population.append(Agent(input_count=config.inputs, layer_sizes=config.hidden_layers, output_count=config.outputs))

    all_time_best_brain = None
    all_time_high_score = -1e9
    
    epoch = 1
    while epoch <= config.total_epochs:       
        generation_banner = stats_font.render(f"Epoch {epoch}", True, (255, 255, 255))
        display_surface.blit(generation_banner, (DISPLAY_W//2 - generation_banner.get_width()//2, 12))
        pygame.display.flip()
        
        fitness_results = run_simulation_cycle(window=display_surface, timer=game_clock, text_renderer=stats_font, brains=population, epoch_idx=epoch)
        
        population, best_brain, best_score = breed_next_gen(current_pop=population, fitness_scores=fitness_results, settings=config, rng=rng_engine) 
        
        if best_score > all_time_high_score:
            all_time_best_brain = best_brain
            all_time_high_score = best_score 
            export_brain_data(all_time_best_brain, "champion_brain.json")
            print("New Champion Saved: champion_brain.json")
            
        print(f"[Epoch {epoch:03d}] Avg Fit={fitness_results.mean():.3f}  Max Fit={fitness_results.max():.3f}  Record={all_time_high_score:.3f}")

        epoch += 1
        
    pygame.quit()
    
if __name__ == "__main__":
    start_evolution_process()