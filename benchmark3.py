import math
import random
import time
import copy
import sys

# 0. 导入设置
try:
    from main import (
        Stone, MCTSAgent, GreedyAgent, resolve_collisions, apply_execution_error, apply_game_rules,
        PHYSICS_WIDTH, PHYSICS_LENGTH, HOG_LINE_Y_START, TEE_Y_DEST, 
        HOUSE_RADIUS, STONE_RADIUS, STONE_YELLOW, STONE_PURPLE, HOG_LINE_Y_DEST,
        Genome
    )
except ImportError:
    print("错误：找不到文件 'main.py' 或导入失败。")
    sys.exit()

# 1. 单局模拟器 (1 End) 
def run_one_end(hammer_color, current_score_diff):
    stones = []
    global_id = 0
    
    # A. 初始化智能体 (Greedy vs Student)
    try:
        my_genome = Genome()
        my_genome.w_center = 104.16   
        my_genome.force_mult = 0.99   
        
        greedy_agent = GreedyAgent(STONE_PURPLE)
        
        student_agent = MCTSAgent(STONE_YELLOW, sim_limit=1000, use_nn=True, genome=my_genome)
        
    except Exception as e:
        print(f"警告：初始化失败 ({e})，正在退出...")
        return 0, 0

    # B. 投壶循环
    for shot in range(16):
        if hammer_color == STONE_YELLOW:
            shooter_color = STONE_PURPLE if shot % 2 == 0 else STONE_YELLOW
        else:
            shooter_color = STONE_YELLOW if shot % 2 == 0 else STONE_PURPLE
            
        is_greedy_turn = (shooter_color == STONE_PURPLE)

        stones_snapshot = copy.deepcopy(stones)

        if is_greedy_turn:
            s, a, sp = greedy_agent.get_action(stones, shot)
        else:
            s, a, sp = student_agent.get_action(stones, shot, current_score_diff)
            
        real_angle = apply_execution_error(a)
            
        global_id += 1
        new_s = Stone(PHYSICS_WIDTH/2, HOG_LINE_Y_START, shooter_color, global_id, sp)
        new_s.vy = s * math.cos(real_angle); new_s.vx = s * math.sin(real_angle)
        stones.append(new_s)
        
        step = 0
        while step < 1500: 
            moving = False
            for _ in range(3):
                for st in stones:
                    if st.vx**2 + st.vy**2 > 0.001: 
                        st.move(dt=0.015 / 3) 
                        moving = True
                resolve_collisions(stones)
            if not moving: break
            step += 1
            
        stones[:] = [st for st in stones if 0 < st.x < PHYSICS_WIDTH and 0 < st.y < PHYSICS_LENGTH + 1.0]
        stones, msg = apply_game_rules(stones_snapshot, stones, shot, shooter_color)

    y_d = sorted([math.sqrt((st.x-PHYSICS_WIDTH/2)**2+(st.y-TEE_Y_DEST)**2) for st in stones if st.color==STONE_YELLOW])
    p_d = sorted([math.sqrt((st.x-PHYSICS_WIDTH/2)**2+(st.y-TEE_Y_DEST)**2) for st in stones if st.color==STONE_PURPLE])
    
    valid_y = [d for d in y_d if d < HOUSE_RADIUS+STONE_RADIUS]
    valid_p = [d for d in p_d if d < HOUSE_RADIUS+STONE_RADIUS]
    
    end_score_y = 0; end_score_p = 0
    
    if valid_y and (not valid_p or valid_y[0] < valid_p[0]):
        limit = valid_p[0] if valid_p else 999
        for d in valid_y: 
            if d < limit: end_score_y += 1
    elif valid_p:
        limit = valid_y[0] if valid_y else 999
        for d in valid_p: 
            if d < limit: end_score_p += 1
            
    return end_score_p, end_score_y

# 2. 整场比赛模拟 (8 Ends)
def run_full_match(match_id):
    total_p = 0; total_y = 0
    hammer = random.choice([STONE_YELLOW, STONE_PURPLE])
    
    print(f"Match {match_id:<2} Start...", end="\r")
    
    for end in range(1, 9):
        score_diff = total_y - total_p
        p_score, y_score = run_one_end(hammer, score_diff)
        total_p += p_score; total_y += y_score
        
        if p_score > 0: hammer = STONE_YELLOW
        elif y_score > 0: hammer = STONE_PURPLE
    
    winner = "YELLOW" if total_y > total_p else ("PURPLE" if total_p > total_y else "DRAW")
    
    print(f"Match {match_id:<2} Result: Greedy {total_p} : {total_y} Student -> Winner: {winner}          ")
    return winner, total_p, total_y

# 3. 主程序入口
if __name__ == "__main__":
    TOTAL_MATCHES = 20
    student_wins = 0
    greedy_wins = 0
    draws = 0
    max_score_diff = 0
    
    start_time = time.time()
    
    print(f"鲁棒性测试: 贪心算法 (Baseline) vs 神经网络Student (Ours)")
    print(" Purple: Greedy Agent (局部最优)")
    print(" Yellow: Student Agent (全局搜索 + NN)")
    
    try:
        for i in range(1, TOTAL_MATCHES + 1):
            winner, p_score, y_score = run_full_match(i)

            if winner == "YELLOW": 
                student_wins += 1
            elif winner == "PURPLE": 
                greedy_wins += 1
            else:
                draws += 1
                
            diff = y_score - p_score
            if diff > max_score_diff:
                max_score_diff = diff
            
    except KeyboardInterrupt:
        print("\n[!] 用户中断。")
        TOTAL_MATCHES = i - 1

    if TOTAL_MATCHES > 0:
        duration = time.time() - start_time
        win_rate = student_wins / TOTAL_MATCHES * 100
        
        print(f"最终测试报告")
        print(f"总场次: {TOTAL_MATCHES} 场 (每场8局)")
        print(f"总耗时: {duration:.1f}s")
        print(f"Student (Yellow) 胜场 : {student_wins}")
        print(f"Greedy (Purple)  胜场 : {greedy_wins}")
        print(f"平局 (Draw)           : {draws}")
        print(f"Student AI 胜率: {win_rate:.1f}%")
        print(f"最大净胜分差: {max_score_diff} 分")