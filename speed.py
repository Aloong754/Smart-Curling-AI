import time
import copy
import math
from main import MCTSAgent, Stone, PHYSICS_WIDTH, HOG_LINE_Y_START, STONE_YELLOW, STONE_PURPLE, resolve_collisions

def run_heavy_physics_simulation(stones):
    new_s = Stone(PHYSICS_WIDTH/2, HOG_LINE_Y_START, STONE_YELLOW, 99)
    new_s.vy = 3.0
    stones.append(new_s)
    
    step = 0
    while step < 1000: 
        moving = False
        for s in stones:
            if s.vx**2 + s.vy**2 > 0.001:
                s.move(dt=0.015)
                moving = True
        
        if not moving: break
        resolve_collisions(stones)
        step += 1
    return len(stones)

class RealPhysicsMCTS(MCTSAgent):
    def _evaluate(self, stones, shot_index):
        run_heavy_physics_simulation(copy.deepcopy(stones))
        return 0.0

def benchmark(agent, name, stones):
    print(f"正在测试 [{name}] (200次思考)...")
    start = time.perf_counter()
    agent.sim_limit = 200
    try:
        agent.get_action(copy.deepcopy(stones), 1, 0)
    except ZeroDivisionError:
        print("警告：检测到物理除零错误，已跳过本次计算")
    end = time.perf_counter()
    duration = end - start
    print(f"   -> 耗时: {duration:.4f} 秒")
    return duration

if __name__ == "__main__":
    stones = []
    for i in range(6):
        offset_x = (i % 2 - 0.5) * 0.5 
        s = Stone(PHYSICS_WIDTH/2 + offset_x, 30.0 + i * 1.0, STONE_YELLOW, i)
        stones.append(s)
    

    agent_nn = MCTSAgent(STONE_YELLOW, use_nn=True)
    t_nn = benchmark(agent_nn, "Neural Agent", stones)

    print(f"正在测试 [Physics Agent] ...")
    
    t_start = time.perf_counter()
    run_heavy_physics_simulation(copy.deepcopy(stones))
    t_one_sim = time.perf_counter() - t_start
    t_phy_total = t_one_sim * 200
    
    print(f"   -> 单次模拟耗时: {t_one_sim*1000:.2f} ms")
    print(f"   -> 200次总耗时(预估): {t_phy_total:.4f} 秒")

    print("实验结论分析:")
    print(f"神经网络 ({t_nn:.2f}s) vs 物理模拟 ({t_phy_total:.2f}s):")
    if t_nn > 0:
        speedup = t_phy_total / t_nn
        print(f"   -> 速度提升: {speedup:.1f} 倍！")