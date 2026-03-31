import pygame
import math
import random
import copy
import numpy as np

# 0. 全局配置
RINK_VIEW_WIDTH = 200   
UI_WIDTH = 500          
SCREEN_WIDTH = RINK_VIEW_WIDTH + UI_WIDTH
SCREEN_HEIGHT = 900
FPS = 60

UI_TOP_MARGIN = 240     
ZOOM_VIEW_HEIGHT = SCREEN_HEIGHT - UI_TOP_MARGIN

PHYSICS_WIDTH = 4.75        
PHYSICS_LENGTH = 44.5       
TEE_Y_DEST = 38.0           
TEE_Y_START = 3.25          
HOG_LINE_Y_DEST = 38.0 - 6.4 
HOG_LINE_Y_START = 3.25 + 6.4 
BACK_LINE_Y_DEST = TEE_Y_DEST + 1.83 
HOUSE_RADIUS = 1.83         
STONE_RADIUS = 0.145

WHITE = (255, 255, 255); BLACK = (0, 0, 0)
RING_BLUE = (0, 80, 200); RING_RED = (220, 20, 20)
STONE_PURPLE = (160, 32, 240); STONE_YELLOW = (255, 220, 0)
GREY = (160, 160, 160); ICE_COLOR = (240, 245, 255)
BG_COLOR = (25, 25, 30); UI_BG_COLOR = (50, 54, 60); TEXT_COLOR = (240, 240, 240)
HOG_LINE_COLOR = (220, 50, 50)
CENTER_LINE_COLOR = (255, 100, 100)

# 1. 核心算法: 手写神经网络
# 遗传算法基因
class Genome:
    def __init__(self):
        self.w_center = 77.07 
        self.w_takeout = 237.29
        self.w_guard = 188.31
        self.w_hammer = 56.48 
        self.force_mult = 1.0633

class ManualNeuralNet:
    def __init__(self, weight_file="model_weights.npz"):
        try:
            data = np.load(weight_file)
            self.W1 = data["W1"]; self.b1 = data["b1"]
            self.W2 = data["W2"]; self.b2 = data["b2"]
            self.W3 = data["W3"]; self.b3 = data["b3"]
        except:
            print("使用随机参数初始化")
            input_size = 34
            self.W1 = np.random.randn(input_size, 256) * 0.1; self.b1 = np.zeros(256)
            self.W2 = np.random.randn(256, 128) * 0.1; self.b2 = np.zeros(128)
            self.W3 = np.random.randn(128, 1) * 0.1; self.b3 = np.zeros(1)

    def relu(self, x): return np.maximum(0, x)
    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x_vector):
        x = np.array(x_vector)
        z1 = np.dot(x, self.W1) + self.b1; a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2; a2 = self.relu(z2)
        z3 = np.dot(a2, self.W3) + self.b3; out = self.sigmoid(z3)
        return out[0]

def state_to_vector(stones, current_color_code, hammer_color_code):
    MAX_W = 4.75
    MAX_L = 44.5
    data = []
    
    sorted_stones = sorted(stones, key=lambda s: s.id)
    
    count = 0
    for s in sorted_stones:
        if count >= 16: break 
        data.append(s.x / MAX_W)
        data.append(s.y / MAX_L)
        count += 1
        
    while len(data) < 16 * 2:
        data.append(0.0)
        
    data.append(current_color_code)
    data.append(hammer_color_code)
    
    return np.array(data)

# 2. 基础逻辑与物理
def apply_execution_error(angle):
    if random.random() < 0.8: return angle
    else: return angle + random.gauss(0, 0.04)

def is_in_house(stone):
    dist = math.sqrt((stone.x-PHYSICS_WIDTH/2)**2 + (stone.y-TEE_Y_DEST)**2)
    return dist < HOUSE_RADIUS + STONE_RADIUS

def is_in_fgz(stone):
    return stone.y > HOG_LINE_Y_DEST and not is_in_house(stone)

def is_touching_center_line(stone):
    return abs(stone.x - PHYSICS_WIDTH/2) < STONE_RADIUS

def apply_game_rules(stones_before, stones_after, shot_index, shooter_color):
    valid_stones = []
    for s in stones_after:
        if 0 < s.x < PHYSICS_WIDTH and s.y < PHYSICS_LENGTH + 0.5: 
            valid_stones.append(s)
    stones_after = valid_stones

    if stones_after:
        shooter = stones_after[-1]
        if shooter.color == shooter_color and shooter.y < HOG_LINE_Y_DEST:
             stones_after.pop() 

    if shot_index < 5:
        protected_map = {s.id: s for s in stones_before if s.color != shooter_color and is_in_fgz(s)}
        current_map = {s.id: s for s in stones_after}
        for pid, old_s in protected_map.items():
            if pid not in current_map: 
                return copy.deepcopy(stones_before), "FOUL! FGZ Violation!"
            if is_touching_center_line(old_s):
                if not is_touching_center_line(current_map[pid]):
                    return copy.deepcopy(stones_before), "FOUL! No-Tick Violation!"
    return stones_after, ""

class Stone:
    def __init__(self, x, y, color, sid=0, angle_vel=0):
        self.x = x; self.y = y
        self.vx = 0.0; self.vy = 0.0
        self.radius = STONE_RADIUS
        self.color = color
        self.id = sid
        self.angle_vel = angle_vel 

    def move(self, dt=0.015, friction_coef=0.025, noise_on=True):
        v = math.sqrt(self.vx**2 + self.vy**2)
        if v < 0.05:
            self.vx, self.vy = 0, 0
            return
        
        if noise_on: noise = random.gauss(1.0, 0.001)
        else: noise = 1.0

        decel = friction_coef * 9.8 * noise
        new_v = max(0, v - decel * dt)
        
        CURL_FACTOR = 0.005 * self.angle_vel * 0.1 
        if v > 0:
            ux = self.vx / v; uy = self.vy / v
            vx_curl = -uy * CURL_FACTOR * dt 
            vy_curl =  ux * CURL_FACTOR * dt
            self.vx = (self.vx / v) * new_v + vx_curl
            self.vy = (self.vy / v) * new_v + vy_curl
        self.x += self.vx * dt
        self.y += self.vy * dt

def resolve_collisions(stones):
    for i in range(len(stones)):
        for j in range(i + 1, len(stones)):
            s1 = stones[i]; s2 = stones[j]
            dx = s1.x - s2.x; dy = s1.y - s2.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < (s1.radius + s2.radius):
                overlap = (s1.radius + s2.radius - dist) / 2
                nx = dx / dist; ny = dy / dist
                s1.x += nx * overlap; s1.y += ny * overlap
                s2.x -= nx * overlap; s2.y -= ny * overlap
                dvx = s1.vx - s2.vx; dvy = s1.vy - s2.vy
                vn = dvx * nx + dvy * ny
                if vn > 0: continue
                impulse = vn 
                s1.vx -= impulse * nx; s1.vy -= impulse * ny
                s2.vx += impulse * nx; s2.vy += impulse * ny
                s1.vx *= 0.98; s1.vy *= 0.98

# 4. 战术生成
class ShotGenerator:
    def __init__(self):
        self.friction = 0.025 
        self.g = 9.8
        
        self.Y_START = HOG_LINE_Y_START
        self.Y_TEE = TEE_Y_DEST
        self.Y_GUARD = HOG_LINE_Y_DEST + 2.0
        
        dist_tee = self.Y_TEE - self.Y_START
        decel = self.friction * self.g
        self.speed_tee = math.sqrt(2 * decel * dist_tee)
        
        dist_guard = self.Y_GUARD - self.Y_START
        self.speed_guard = math.sqrt(2 * decel * dist_guard)
        
        self.speed_hit = self.speed_tee * 1.6
        self.speed_freeze = self.speed_tee * 0.98

        # print(f"校准完成: Tee速度={self.speed_tee:.2f}, Guard速度={self.speed_guard:.2f}")

    def _get_safe_guard_action(self, stones):
        candidates = [0, -0.6, 0.6, -1.0, 1.0]
        target_y = self.Y_GUARD
        for off in candidates:
            target_x = PHYSICS_WIDTH/2 + off
            is_occupied = False
            for s in stones:
                if abs(s.y - target_y) < 1.5 and abs(s.x - target_x) < 0.8:
                    is_occupied = True; break
            if not is_occupied:
                angle = math.atan2(off, target_y - self.Y_START)
                return [(self.speed_guard, angle, 0), (self.speed_guard, angle, 1)]
        return [(self.speed_guard, 0.05, 0)]

    def _get_freeze_action(self, stones, my_color):
        actions = []
        in_house_stones = [s for s in stones if is_in_house(s)]
        
        for target in in_house_stones:
            dx = target.x - PHYSICS_WIDTH/2
            dy = target.y - self.Y_START
            angle = math.atan2(dx, dy)
            
            dist = math.sqrt(dx**2 + dy**2)
            decel = self.friction * self.g
            perfect_speed = math.sqrt(2 * decel * dist)
            
            actions.append((perfect_speed, angle, 0))
        return actions

    def _check_path(self, target_s, all_stones):
        x1, y1 = PHYSICS_WIDTH/2, self.Y_START
        x2, y2 = target_s.x, target_s.y
        blocked = False; left_blocked = False; right_blocked = False
        for s in all_stones:
            if s.id == target_s.id: continue
            if y1 < s.y < y2:
                ideal_x = x1 + (x2 - x1) * (s.y - y1) / (y2 - y1)
                dist = s.x - ideal_x
                if abs(dist) < STONE_RADIUS * 2.2:
                    blocked = True
                    if dist < 0: left_blocked = True 
                    else: right_blocked = True 
        if not blocked: return (False, 0)
        if left_blocked and not right_blocked: return (True, -1)
        if right_blocked and not left_blocked: return (True, 1)
        return (True, 0)

    def get_strategy_candidates(self, stones, my_color, shot_index, score_diff):
        actions = []
        
        # 1. 开局前4壶，优先占位
        if shot_index < 4:
            actions.extend(self._get_safe_guard_action(stones))
            actions.append((self.speed_tee, 0, 0)) 
            return actions

        # 2. 进攻与击打
        opps = [s for s in stones if s.color != my_color]
        if opps:
            target = min(opps, key=lambda s: math.sqrt((s.x-PHYSICS_WIDTH/2)**2+(s.y-TEE_Y_DEST)**2))
            blocked, side = self._check_path(target, stones)
            dx = target.x - PHYSICS_WIDTH/2; dy = target.y - self.Y_START
            base_angle = math.atan2(dx, dy)
            
            if not blocked: 
                actions.append((self.speed_hit, base_angle, 0))
                actions.append((self.speed_hit * 0.8, base_angle, 0))
            else:
                if side == -1: actions.append((self.speed_tee * 1.05, base_angle + 0.08, -1))
                elif side == 1: actions.append((self.speed_tee * 1.05, base_angle - 0.08, 1))
            
            actions.extend(self._get_freeze_action(stones, my_color))

        actions.append((self.speed_tee, 0, 0))
        actions.append((self.speed_tee, 0.01, 0))
        actions.append((self.speed_tee, -0.01, 0))
        
        if shot_index == 15:
            actions.append((self.speed_hit * 1.2, 0, 0))
        
        return actions

# 3. MCTS智能体
class MCTSNode:
    def __init__(self, stones, parent=None, action=None):
        self.stones = stones; self.parent = parent
        self.children = []; self.visits = 0; self.wins = 0.0; self.action = action
    def ucb1(self, explore=2.0):
        if self.visits == 0: return float('inf')
        return (self.wins/self.visits) + explore * math.sqrt(math.log(self.parent.visits)/self.visits)

class MCTSAgent:
    def __init__(self, my_color, sim_limit=600, use_nn=False, genome=None):
        self.my_color = my_color
        self.sim_limit = sim_limit
        self.generator = ShotGenerator()
        self.id_counter = 100 
        self.use_nn = use_nn
        self.genome = genome if genome else Genome()
        
        # 初始化手写神经网络
        self.net = None
        if self.use_nn:
            self.net = ManualNeuralNet(weight_file="model_weights.npz")

    def get_action(self, current_stones, shot_index, score_diff=0):
        root = MCTSNode(copy.deepcopy(current_stones))
        for _ in range(self.sim_limit):
            node = self._select(root)
            if not self._is_terminal(node): 
                node = self._expand(node, shot_index, score_diff) 
            score = self._simulate(node, shot_index, score_diff)
            self._backpropagate(node, score)
        
        if not root.children: 
            return (self.generator.speed_tee, 0, 0)
        return max(root.children, key=lambda c: c.visits).action

    def _select(self, node):
        while len(node.children) > 0: 
            node = max(node.children, key=lambda c: c.ucb1())
        return node

    def _expand(self, node, shot_index, score_diff):
        tried = [c.action for c in node.children]
        candidates = self.generator.get_strategy_candidates(node.stones, self.my_color, shot_index, score_diff)
        candidates = list(set(candidates))
        random.shuffle(candidates)
        
        for action in candidates:
            if action not in tried:
                next_stones = self._mock_play(node.stones, action, shot_index)
                if next_stones is None: 
                    child = MCTSNode([], parent=node, action=action)
                    child.wins = -99999; child.visits = 1
                    node.children.append(child)
                else:
                    child = MCTSNode(next_stones, parent=node, action=action)
                    node.children.append(child)
                return child
        return node

    def _simulate(self, node, shot_index, score_diff):
        if not node.stones: return -99999
        return self._evaluate(node.stones, shot_index)

    def _evaluate(self, stones, shot_index):
        # 1. 神经网络评分
        if self.use_nn and self.net:
            my_color_code = 1.0 if self.my_color == STONE_YELLOW else -1.0
            inp = state_to_vector(stones, my_color_code, 1.0)
            win_prob = self.net.forward(inp)
            return (win_prob - 0.5) * 5000

        # 2. 启发式评分
        my = [s for s in stones if s.color == self.my_color]
        opp = [s for s in stones if s.color != self.my_color]
        score = 0.0
        for s in my:
            if is_in_house(s):
                dist = math.sqrt((s.x-PHYSICS_WIDTH/2)**2+(s.y-TEE_Y_DEST)**2)
                score += max(0, 150 - dist * 60)
        for s in opp:
            if is_in_house(s):
                score -= 100 
        return score

    def _backpropagate(self, node, score):
        while node:
            node.visits += 1; node.wins += score; node = node.parent

    def _is_terminal(self, node): 
        return len(node.stones) >= 16

    def _mock_play(self, stones, action, shot_index):
        temp = copy.deepcopy(stones); bk = copy.deepcopy(temp)
        self._apply_action(temp, action)
        self._fast_forward(temp)
        final, msg = apply_game_rules(bk, temp, shot_index, self.my_color)
        if msg: return None 
        return final

    def _apply_action(self, stones, action):
        s, a, spin = action
        s = s * self.genome.force_mult 
        self.id_counter += 1
        new_s = Stone(PHYSICS_WIDTH/2, HOG_LINE_Y_START, self.my_color, self.id_counter, angle_vel=spin)
        new_s.vy = s * math.cos(a); new_s.vx = s * math.sin(a)
        stones.append(new_s)

    def _fast_forward(self, stones):
        step = 0
        while step < 2500: 
            moving = False
            for s in stones:
                 if s.vx**2 + s.vy**2 > 0.001: 
                     s.move(dt=0.015) 
                     moving = True
            if not moving: break
            resolve_collisions(stones)
            step += 1
        stones[:] = [s for s in stones if 0 < s.x < PHYSICS_WIDTH and 0 < s.y < PHYSICS_LENGTH + 1.0]

class GreedyAgent:
    def __init__(self, my_color): 
        self.my_color = my_color; self.gen = ShotGenerator()
    def get_action(self, stones, shot_index):
        targets = [s for s in stones if s.color != self.my_color]
        if shot_index < 5: targets = [t for t in targets if not is_in_fgz(t)]
        if targets:
            t = sorted(targets, key=lambda x: x.y)[0]
            dx = t.x - PHYSICS_WIDTH/2; dy = t.y - HOG_LINE_Y_START
            return (self.gen.speed_hit, math.atan2(dx, dy), 0)
        return (self.gen.speed_tee, random.uniform(-0.005, 0.005), 0)

# 4. UI 绘制
def draw_scene(screen, stones, g_data, prediction=None):
    screen.fill(BG_COLOR)
    
    scale_full_y = SCREEN_HEIGHT / PHYSICS_LENGTH
    scale_full_x = RINK_VIEW_WIDTH / PHYSICS_WIDTH
    scale_full = min(scale_full_x, scale_full_y)
    offset_full_x = (RINK_VIEW_WIDTH - PHYSICS_WIDTH * scale_full) / 2
    offset_full_y = (SCREEN_HEIGHT - PHYSICS_LENGTH * scale_full) / 2

    def to_screen_full(px, py):
        sx = offset_full_x + px * scale_full
        sy = SCREEN_HEIGHT - (offset_full_y + py * scale_full)
        return int(sx), int(sy)

    rink_rect = pygame.Rect(offset_full_x, offset_full_y, PHYSICS_WIDTH * scale_full, PHYSICS_LENGTH * scale_full)
    pygame.draw.rect(screen, ICE_COLOR, rink_rect)
    cx_full = to_screen_full(PHYSICS_WIDTH/2, 0)[0]
    pygame.draw.line(screen, CENTER_LINE_COLOR, (cx_full, rink_rect.top), (cx_full, rink_rect.bottom), 1)

    def draw_house_at(py_center):
        c_scr = to_screen_full(PHYSICS_WIDTH/2, py_center)
        r_scr = int(HOUSE_RADIUS * scale_full)
        pygame.draw.circle(screen, RING_BLUE, c_scr, r_scr)
        pygame.draw.circle(screen, WHITE, c_scr, int(r_scr * 0.66))
        pygame.draw.circle(screen, RING_RED, c_scr, int(r_scr * 0.33))

    draw_house_at(TEE_Y_DEST)
    draw_house_at(TEE_Y_START)

    back_y = to_screen_full(0, BACK_LINE_Y_DEST)[1]
    pygame.draw.line(screen, GREY, (rink_rect.left, back_y), (rink_rect.right, back_y), 1)
    hog_dest_y = to_screen_full(0, HOG_LINE_Y_DEST)[1]
    pygame.draw.line(screen, HOG_LINE_COLOR, (rink_rect.left, hog_dest_y), (rink_rect.right, hog_dest_y), 2)
    hog_start_y = to_screen_full(0, HOG_LINE_Y_START)[1]
    pygame.draw.line(screen, HOG_LINE_COLOR, (rink_rect.left, hog_start_y), (rink_rect.right, hog_start_y), 2)

    r_stone_full = max(2, int(STONE_RADIUS * scale_full))
    for s in stones:
        pos = to_screen_full(s.x, s.y)
        pygame.draw.circle(screen, s.color, pos, r_stone_full)
        pygame.draw.circle(screen, BLACK, pos, r_stone_full, 1)

    if prediction:
        pred_x, pred_y = prediction
        start_pos = to_screen_full(PHYSICS_WIDTH/2, HOG_LINE_Y_START)
        end_pos = to_screen_full(pred_x, pred_y)
        pygame.draw.line(screen, (50, 255, 50), start_pos, end_pos, 1) 
        pygame.draw.circle(screen, (150, 50, 255), end_pos, r_stone_full, 1) 
        pygame.draw.line(screen, (255, 50, 50), (end_pos[0]-3, end_pos[1]-3), (end_pos[0]+3, end_pos[1]+3), 2)
        pygame.draw.line(screen, (255, 50, 50), (end_pos[0]-3, end_pos[1]+3), (end_pos[0]+3, end_pos[1]-3), 2)

    VIEW_Y_MIN = HOG_LINE_Y_DEST - 2.0 
    VIEW_Y_MAX = BACK_LINE_Y_DEST + 1.0
    view_height_physics = VIEW_Y_MAX - VIEW_Y_MIN
    scale_zoom = ZOOM_VIEW_HEIGHT / view_height_physics
    view_center_px = PHYSICS_WIDTH / 2
    view_center_py = (VIEW_Y_MIN + VIEW_Y_MAX) / 2
    
    screen_zoom_cx = RINK_VIEW_WIDTH + UI_WIDTH / 2
    screen_zoom_cy = UI_TOP_MARGIN + ZOOM_VIEW_HEIGHT / 2 

    def to_screen_zoom(px, py):
        sx = screen_zoom_cx + (px - view_center_px) * scale_zoom
        sy = screen_zoom_cy - (py - view_center_py) * scale_zoom
        return int(sx), int(sy)

    zoom_area_rect = pygame.Rect(RINK_VIEW_WIDTH, UI_TOP_MARGIN, UI_WIDTH, ZOOM_VIEW_HEIGHT)
    pygame.draw.rect(screen, ICE_COLOR, zoom_area_rect)
    pygame.draw.line(screen, BLACK, (RINK_VIEW_WIDTH, UI_TOP_MARGIN), (SCREEN_WIDTH, UI_TOP_MARGIN), 2)
    
    cz_top = to_screen_zoom(PHYSICS_WIDTH/2, VIEW_Y_MAX)
    cz_bot = to_screen_zoom(PHYSICS_WIDTH/2, VIEW_Y_MIN)
    pygame.draw.line(screen, CENTER_LINE_COLOR, cz_top, cz_bot, 1)
    
    hz_l = to_screen_zoom(0, HOG_LINE_Y_DEST); hz_r = to_screen_zoom(PHYSICS_WIDTH, HOG_LINE_Y_DEST)
    pygame.draw.line(screen, HOG_LINE_COLOR, hz_l, hz_r, 3)
    
    bz_l = to_screen_zoom(0, BACK_LINE_Y_DEST); bz_r = to_screen_zoom(PHYSICS_WIDTH, BACK_LINE_Y_DEST)
    pygame.draw.line(screen, GREY, bz_l, bz_r, 2)

    center_zoom = to_screen_zoom(PHYSICS_WIDTH/2, TEE_Y_DEST)
    r_zoom_base = int(HOUSE_RADIUS * scale_zoom)
    pygame.draw.circle(screen, RING_BLUE, center_zoom, r_zoom_base)
    pygame.draw.circle(screen, WHITE, center_zoom, int(r_zoom_base * 0.66))
    pygame.draw.circle(screen, RING_RED, center_zoom, int(r_zoom_base * 0.33))

    r_stone_zoom = int(STONE_RADIUS * scale_zoom)
    for s in stones:
        if VIEW_Y_MIN < s.y < VIEW_Y_MAX:
            pos_z = to_screen_zoom(s.x, s.y)
            pygame.draw.circle(screen, (50,50,50, 80), (pos_z[0]+4, pos_z[1]+4), r_stone_zoom) 
            pygame.draw.circle(screen, s.color, pos_z, r_stone_zoom)
            pygame.draw.circle(screen, BLACK, pos_z, r_stone_zoom, 2)
            pygame.draw.circle(screen, (255,255,255, 100), (pos_z[0]-r_stone_zoom//3, pos_z[1]-r_stone_zoom//3), r_stone_zoom//3)

    font_title = pygame.font.SysFont("Arial", 28, bold=True)
    font_info = pygame.font.SysFont("Arial", 22)
    ui_x = RINK_VIEW_WIDTH + 20; ui_y = 20

    title = font_title.render(g_data.get('title', "Curling Game"), True, WHITE)
    screen.blit(title, (ui_x, ui_y))
    
    p_name = g_data.get('p_name', "Purple")
    y_name = g_data.get('y_name', "Yellow")
    score_str = f"{p_name}: {g_data['score_p']}   {y_name}: {g_data['score_y']}"
    score_surf = font_title.render(score_str, True, (255, 220, 0))
    screen.blit(score_surf, (ui_x, ui_y + 40))

    info_str = f"End: {g_data['end']}   Shot: {g_data['shot']}/16"
    info_surf = font_info.render(info_str, True, GREY)
    screen.blit(info_surf, (ui_x, ui_y + 80))

    if g_data['turn'] == "STUDENT":
        turn_txt = f"Turn: {y_name} (AI)"
        turn_col = STONE_YELLOW
    elif g_data['turn'] == "HUMAN":
        turn_txt = f"Turn: {p_name} (You)"
        turn_col = STONE_PURPLE
    else: 
        turn_txt = f"Turn: {p_name} (AI)"
        turn_col = STONE_PURPLE
        
    turn_surf = font_info.render(turn_txt, True, turn_col)
    screen.blit(turn_surf, (ui_x, ui_y + 110))

    state_str = f"State: {g_data['state']}"
    state_surf = font_info.render(state_str, True, WHITE)
    screen.blit(state_surf, (ui_x, ui_y + 140))

    if g_data['msg']:
        msg_surf = font_title.render(g_data['msg'], True, (255, 50, 50))
        screen.blit(msg_surf, (ui_x + 250, ui_y + 140))

# 5. 主程序
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Curling AI System")
    clock = pygame.time.Clock()

    best_genome = Genome()
    best_genome.w_center = 104.16 
    best_genome.force_mult = 0.99 

    print("请选择模式:")
    print("1. Greedy vs Student")
    print("2. Human vs Student)")
    
    raw_input = input("请输入序号 (1/2): ").strip()
    
    if raw_input == '1':
        mode = '1'
    else:
        print(f"输入识别为: '{raw_input}'，切换为: 2 (人机对战)")
        mode = '2'

    student_agent = MCTSAgent(STONE_YELLOW, sim_limit=2000, use_nn=True, genome=best_genome)
    opponent_agent = None
    ui_config = {}

    if mode == '1':
        opponent_agent = GreedyAgent(STONE_PURPLE)
        opponent_name = "Greedy"
        ui_config = {'title': "Exp 1: Student vs Greedy", 'p_name': "Greedy", 'y_name': "Student"}
    else:
        opponent_agent = None 
        opponent_name = "Human" 
        ui_config = {'title': "Demo: Human vs Student AI", 'p_name': "Human", 'y_name': "Student"}
        pygame.display.set_caption(ui_config['title'])

    g_data = {'end': 1, 'shot': 0, 'score_p': 0, 'score_y': 0,
              'hammer': STONE_YELLOW, 'turn': opponent_name.upper(), 'state': "IDLE", 'msg': ""}
    g_data.update(ui_config)

    stones = []; global_id = 0
    cur_first = STONE_PURPLE if g_data['hammer'] == STONE_YELLOW else STONE_YELLOW
    current_actor_color = cur_first

    wait_timer = 0
    stones_snapshot = []
    
    mouse_dragging = False; mouse_start_pos = (0, 0); current_prediction = None

    while True:
        dt = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return
            
            if mode == '2' and g_data['state'] == "IDLE" and current_actor_color == STONE_PURPLE:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_dragging = True; mouse_start_pos = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONUP:
                    if mouse_dragging:
                        mouse_dragging = False; current_prediction = None
                        end_pos = pygame.mouse.get_pos()
                        dx = mouse_start_pos[0] - end_pos[0]; dy = mouse_start_pos[1] - end_pos[1]
                        dist = math.sqrt(dx**2 + dy**2)
                        if dist > 10: 
                            speed = min(max(dist / 25.0, 0.5), 6.0) 
                            angle = math.atan2(dx, dy)
                            stones_snapshot = copy.deepcopy(stones)
                            global_id += 1
                            new_s = Stone(PHYSICS_WIDTH/2, HOG_LINE_Y_START, STONE_PURPLE, global_id, 0)
                            new_s.vy = speed * math.cos(angle); new_s.vx = speed * math.sin(angle)
                            stones.append(new_s)
                            g_data['state'] = "MOVING"; g_data['shot'] += 1; wait_timer = 0

        if mode == '2' and mouse_dragging:
            mx, my = pygame.mouse.get_pos()
            dx = mouse_start_pos[0] - mx; dy = mouse_start_pos[1] - my
            dist_px = math.sqrt(dx**2 + dy**2)
            speed = min(max(dist_px / 25.0, 0.5), 6.0)
            angle = math.atan2(dx, dy)
            friction = 0.025; decel = friction * 9.8
            stop_dist = (speed**2) / (2 * decel)
            pred_x = PHYSICS_WIDTH/2 + stop_dist * math.sin(angle)
            pred_y = HOG_LINE_Y_START + stop_dist * math.cos(angle)
            current_prediction = (pred_x, pred_y)
        else: current_prediction = None

        if g_data['state'] == "IDLE":
            wait_timer += dt
            is_human_turn = (mode == '2' and current_actor_color == STONE_PURPLE)
            
            if wait_timer > 500 and not is_human_turn: 
                wait_timer = 0; stones_snapshot = copy.deepcopy(stones)
                score_diff = g_data['score_y'] - g_data['score_p']
                
                if current_actor_color == STONE_PURPLE:
                    g_data['turn'] = opponent_name.upper()
                    if opponent_agent: 
                        s, a, sp = opponent_agent.get_action(stones, g_data['shot'])
                    else:
                        continue 
                else:
                    g_data['turn'] = "STUDENT"
                    s, a, sp = student_agent.get_action(stones, g_data['shot'], score_diff)
                
                real_angle = apply_execution_error(a)
                global_id += 1
                new_s = Stone(PHYSICS_WIDTH/2, HOG_LINE_Y_START, current_actor_color, global_id, sp)
                new_s.vy = s * math.cos(real_angle); new_s.vx = s * math.sin(real_angle)
                stones.append(new_s)
                g_data['state'] = "MOVING"; g_data['shot'] += 1

        elif g_data['state'] == "MOVING":
            moving = False
            for _ in range(3):
                for s in stones:
                     if s.vx**2 + s.vy**2 > 0.001: s.move(dt=0.015 / 3); moving = True
                resolve_collisions(stones) 
            stones[:] = [s for s in stones if 0 < s.x < PHYSICS_WIDTH and 0 < s.y < PHYSICS_LENGTH + 1.0]

            if not moving:
                shooter_col = current_actor_color
                stones, msg = apply_game_rules(stones_snapshot, stones, g_data['shot']-1, shooter_col)
                g_data['msg'] = msg
                if g_data['shot'] >= 16: g_data['state'] = "END_OVER"
                else:
                    g_data['state'] = "IDLE"
                    current_actor_color = STONE_YELLOW if current_actor_color == STONE_PURPLE else STONE_PURPLE
                    if mode == '2' and current_actor_color == STONE_PURPLE: g_data['turn'] = "HUMAN"
                    elif current_actor_color == STONE_PURPLE: g_data['turn'] = opponent_name.upper()
                    else: g_data['turn'] = "STUDENT"

        elif g_data['state'] == "END_OVER":
            y_stones = [s for s in stones if s.color == STONE_YELLOW and is_in_house(s)]
            p_stones = [s for s in stones if s.color == STONE_PURPLE and is_in_house(s)]
            y_dists = sorted([math.sqrt((s.x-PHYSICS_WIDTH/2)**2+(s.y-TEE_Y_DEST)**2) for s in y_stones])
            p_dists = sorted([math.sqrt((s.x-PHYSICS_WIDTH/2)**2+(s.y-TEE_Y_DEST)**2) for s in p_stones])
            
            pts_y, pts_p = 0, 0
            if y_dists and (not p_dists or y_dists[0] < p_dists[0]):
                thresh = p_dists[0] if p_dists else 999.0
                for d in y_dists: 
                    if d < thresh: pts_y += 1
                g_data['msg'] = f"Yellow Wins End! +{pts_y}"
            elif p_dists:
                thresh = y_dists[0] if y_dists else 999.0
                for d in p_dists: 
                    if d < thresh: pts_p += 1
                g_data['msg'] = f"Purple Wins End! +{pts_p}"
            else:
                g_data['msg'] = "Blank End"

            g_data['score_y'] += pts_y
            g_data['score_p'] += pts_p
            
            draw_scene(screen, stones, g_data, current_prediction)
            pygame.display.flip()
            pygame.time.delay(3000)

            g_data['end'] += 1; g_data['shot'] = 0; g_data['state'] = "IDLE"; stones = []
            if pts_y > 0: g_data['hammer'] = STONE_PURPLE
            elif pts_p > 0: g_data['hammer'] = STONE_YELLOW
            cur_first = STONE_PURPLE if g_data['hammer'] == STONE_YELLOW else STONE_YELLOW
            current_actor_color = cur_first
            
            if mode == '2' and current_actor_color == STONE_PURPLE: g_data['turn'] = "HUMAN"
            elif current_actor_color == STONE_PURPLE: g_data['turn'] = opponent_name.upper()
            else: g_data['turn'] = "STUDENT"
            g_data['msg'] = ""

        draw_scene(screen, stones, g_data, current_prediction)
        
        if mode == '2' and mouse_dragging:
            mx, my = pygame.mouse.get_pos()
            pygame.draw.line(screen, (255, 255, 0), mouse_start_pos, (mx, my), 2)
            font = pygame.font.SysFont("Arial", 20)
            dx = mouse_start_pos[0] - mx; dy = mouse_start_pos[1] - my
            dist = math.sqrt(dx**2 + dy**2)
            power_val = min(max(dist / 25.0, 0.5), 6.0)
            txt = font.render(f"Power: {power_val:.1f} m/s", True, (255, 50, 50))
            screen.blit(txt, (mx + 20, my))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()