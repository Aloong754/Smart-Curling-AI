import random
import copy
import math
import numpy as np
import sys

try:
    from main import (
        Genome, MCTSAgent, Stone, resolve_collisions, apply_game_rules,
        is_in_house, TEE_Y_DEST, PHYSICS_WIDTH, HOG_LINE_Y_START,
        STONE_YELLOW, STONE_PURPLE
    )
    print("成功加载 main.py 核心模块")
except ImportError:
    print("错误：找不到 main.py")
    sys.exit()


POP_SIZE = 10 
GENERATIONS = 5   
SIM_ENDS = 1         

def create_random_genome():
    g = Genome()
    g.w_center = random.uniform(50, 150)
    g.w_takeout = random.uniform(100, 300)
    g.w_guard = random.uniform(50, 200)
    g.w_hammer = random.uniform(10, 100)
    g.force_mult = random.uniform(0.9, 1.1)
    return g

def mutate(g):
    if random.random() < 0.3: g.w_center += random.gauss(0, 10)
    if random.random() < 0.3: g.w_takeout += random.gauss(0, 10)
    if random.random() < 0.3: g.w_guard += random.gauss(0, 10)
    if random.random() < 0.3: g.w_hammer += random.gauss(0, 5)
    if random.random() < 0.3: g.force_mult += random.gauss(0, 0.02)
    g.force_mult = max(0.8, min(1.2, g.force_mult))
    return g

def crossover(g1, g2):
    child = Genome()
    child.w_center = random.choice([g1.w_center, g2.w_center])
    child.w_takeout = random.choice([g1.w_takeout, g2.w_takeout])
    child.w_guard = random.choice([g1.w_guard, g2.w_guard])
    child.w_hammer = random.choice([g1.w_hammer, g2.w_hammer])
    child.force_mult = (g1.force_mult + g2.force_mult) / 2
    return child

def simulate_game(genome_a, genome_b):

    agent_a = MCTSAgent(STONE_YELLOW, sim_limit=50, use_nn=True, genome=genome_a)
    agent_b = MCTSAgent(STONE_PURPLE, sim_limit=50, use_nn=True, genome=genome_b)
    
    total_score_diff = 0

    for end in range(SIM_ENDS):
        stones = []

        hammer_color = STONE_PURPLE 
        turn_order = [STONE_YELLOW, STONE_PURPLE] * 8
        
        shot_count = 0
        current_id = 0
        
        for turn_color in turn_order:
            if turn_color == STONE_YELLOW:
                speed, angle, spin = agent_a.get_action(stones, shot_count, 0)
                force = speed * genome_a.force_mult 
            else:
                speed, angle, spin = agent_b.get_action(stones, shot_count, 0)
                force = speed * genome_b.force_mult

            current_id += 1
            new_s = Stone(PHYSICS_WIDTH/2, HOG_LINE_Y_START, turn_color, current_id, angle_vel=spin)
            new_s.vy = force * math.cos(angle)
            new_s.vx = force * math.sin(angle)
            stones.append(new_s)

            step = 0
            while step < 1500:
                moving = False
                for s in stones:
                    if s.vx**2 + s.vy**2 > 0.001:
                        s.move(dt=0.015)
                        moving = True
                
                if not moving: break
                
                resolve_collisions(stones)
                step += 1
            
            shooter = stones[-1]
            stones, msg = apply_game_rules(stones, stones, shot_count, turn_color)
            shot_count += 1
            
        y_stones = [s for s in stones if s.color == STONE_YELLOW and is_in_house(s)]
        p_stones = [s for s in stones if s.color == STONE_PURPLE and is_in_house(s)]
        
        y_dists = sorted([math.sqrt((s.x-PHYSICS_WIDTH/2)**2+(s.y-TEE_Y_DEST)**2) for s in y_stones])
        p_dists = sorted([math.sqrt((s.x-PHYSICS_WIDTH/2)**2+(s.y-TEE_Y_DEST)**2) for s in p_stones])
        
        pts_a = 0
        pts_b = 0
        
        if y_dists and (not p_dists or y_dists[0] < p_dists[0]):
            thresh = p_dists[0] if p_dists else 999.0
            for d in y_dists: 
                if d < thresh: pts_a += 1
        elif p_dists:
            thresh = y_dists[0] if y_dists else 999.0
            for d in p_dists: 
                if d < thresh: pts_b += 1
        
        total_score_diff += (pts_a - pts_b)
        
    return total_score_diff


def train():
    print(f"开始进化训练 (Population={POP_SIZE}, Gens={GENERATIONS})...")
    
    population = [create_random_genome() for _ in range(POP_SIZE)]
    
    for gen in range(GENERATIONS):
        scores = [0] * POP_SIZE
        print(f"\nGeneration {gen+1} / {GENERATIONS}")
        
        for i in range(POP_SIZE):
            opp_idx = (i + 1) % POP_SIZE
            
            print(f"\r对战: Agent {i} vs Agent {opp_idx} ...", end="")
            
            diff = simulate_game(population[i], population[opp_idx])
            
            if diff > 0: scores[i] += 3 
            elif diff == 0: scores[i] += 1 
            else: scores[opp_idx] += 3  
            
        print("\n本代对战结束。")
            
        best_idx = scores.index(max(scores))
        best_g = population[best_idx]
        print(f"Best Agent ID: {best_idx}, Score: {scores[best_idx]}")
        print(f"Genes: Center={best_g.w_center:.1f}, Takeout={best_g.w_takeout:.1f}, Force={best_g.force_mult:.3f}")

        new_pop = [copy.deepcopy(best_g)]
        
        while len(new_pop) < POP_SIZE:
            p1 = population[best_idx]
            p2 = random.choice(population)
            
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
            
        population = new_pop

    return population[0]

if __name__ == "__main__":
    best_genome = train()
    
    print("进化完成！最佳基因参数：")
    print(f"self.w_center = {best_genome.w_center:.2f}")
    print(f"self.w_takeout = {best_genome.w_takeout:.2f}")
    print(f"self.w_guard = {best_genome.w_guard:.2f}")
    print(f"self.w_hammer = {best_genome.w_hammer:.2f}")
    print(f"self.force_mult = {best_genome.force_mult:.4f}")
