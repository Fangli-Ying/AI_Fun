import random
import sys
import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt

# ========= 游戏配置 =========
SW = 400  # 屏幕宽度
SH = 600  # 屏幕高度
FPS = 60
GRAVITY = 0.15  # 重力加速度
THRUST_POWER = 0.3  # 推进器力量
ROTATION_SPEED = 2  # 旋转速度
FUEL_CONSUMPTION = 1  # 燃料消耗

# ========= 初始化 =========
pygame.init()
WINDOW = pygame.display.set_mode((SW, SH))
pygame.display.set_caption("Mars Lander AI")
FPSCLOCK = pygame.time.Clock()
Font = pygame.font.Font(None, 28)
BigFont = pygame.font.Font(None, 40)

# ========= 颜色定义 =========
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (50, 255, 50)
MARS_BROWN = (139, 69, 19)
MARS_DARK = (101, 67, 33)
MARS_LIGHT = (160, 82, 45)
GRAY = (128, 128, 128)
BLUE = (100, 149, 237)

# ========= Q-learning 参数 =========
# 状态空间：[高度区间, 垂直速度区间, 水平速度区间, 角度区间, 燃料区间]
HEIGHT_BINS = 8      # 高度分8档
VEL_Y_BINS = 7       # 垂直速度分7档
VEL_X_BINS = 5       # 水平速度分5档
ANGLE_BINS = 7       # 角度分7档
FUEL_BINS = 3        # 燃料分3档

# 动作：0=无操作, 1=主推进器, 2=左转, 3=右转
NUM_ACTIONS = 4
Q = np.zeros((HEIGHT_BINS, VEL_Y_BINS, VEL_X_BINS, ANGLE_BINS, FUEL_BINS, NUM_ACTIONS))

# 学习参数
ALPHA = 0.1  # 学习率
GAMMA = 0.95  # 折扣因子
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.997

# ========= 地形生成 =========
def generate_terrain():
    """生成不平整的火星地形"""
    terrain_points = []
    segment_width = 10
    num_segments = SW // segment_width
    
    # 基础高度
    base_height = SH - 80
    
    # 生成随机起伏
    for i in range(num_segments + 1):
        x = i * segment_width
        # 使用正弦波 + 随机噪声生成起伏
        noise = random.randint(-30, 30)
        wave = int(20 * np.sin(i * 0.3))
        y = base_height + noise + wave
        terrain_points.append((x, y))
    
    # 确保着陆区域相对平坦（中间区域）
    landing_zone_start = num_segments // 2 - 3
    landing_zone_end = num_segments // 2 + 3
    landing_y = terrain_points[landing_zone_start][1]
    
    for i in range(landing_zone_start, landing_zone_end + 1):
        x = i * segment_width
        # 着陆区域稍微平坦，但仍有些许起伏
        terrain_points[i] = (x, landing_y + random.randint(-5, 5))
    
    return terrain_points, landing_zone_start * segment_width, landing_zone_end * segment_width

# ========= 飞船类 =========
class Lander:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.x = SW // 2
        self.y = 50
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = 0
        self.angle = random.uniform(-15, 15)  # 初始角度
        self.fuel = 100
        self.crashed = False
        self.landed = False
    
    def get_state(self):
        """将连续状态离散化为Q表索引"""
        # 高度
        height_idx = min(HEIGHT_BINS - 1, int(self.y / (SH / HEIGHT_BINS)))
        
        # 垂直速度
        vy_clipped = max(-5, min(5, self.vy))
        vy_idx = int((vy_clipped + 5) / 10 * (VEL_Y_BINS - 1))
        vy_idx = max(0, min(VEL_Y_BINS - 1, vy_idx))
        
        # 水平速度
        vx_clipped = max(-3, min(3, self.vx))
        vx_idx = int((vx_clipped + 3) / 6 * (VEL_X_BINS - 1))
        vx_idx = max(0, min(VEL_X_BINS - 1, vx_idx))
        
        # 角度
        angle_clipped = max(-45, min(45, self.angle))
        angle_idx = int((angle_clipped + 45) / 90 * (ANGLE_BINS - 1))
        angle_idx = max(0, min(ANGLE_BINS - 1, angle_idx))
        
        # 燃料
        fuel_idx = min(FUEL_BINS - 1, int(self.fuel / 100 * FUEL_BINS))
        
        return (height_idx, vy_idx, vx_idx, angle_idx, fuel_idx)
    
    def apply_action(self, action):
        """应用动作并返回奖励"""
        reward = -1  # 每步小惩罚，鼓励快速着陆
        
        # 动作执行
        thrust_main = (action == 1)
        thrust_left = (action == 2)
        thrust_right = (action == 3)
        
        # 重力
        self.vy += GRAVITY
        
        # 主推进器（向下喷射，产生向上推力）
        if thrust_main and self.fuel > 0:
            rad = np.radians(self.angle)
            self.vx -= np.sin(rad) * THRUST_POWER
            self.vy -= np.cos(rad) * THRUST_POWER
            self.fuel -= FUEL_CONSUMPTION
        
        # 左转推进器
        if thrust_left and self.fuel > 0:
            self.angle -= ROTATION_SPEED
            self.fuel -= FUEL_CONSUMPTION * 0.5
        
        # 右转推进器
        if thrust_right and self.fuel > 0:
            self.angle += ROTATION_SPEED
            self.fuel -= FUEL_CONSUMPTION * 0.5
        
        # 更新位置
        self.x += self.vx
        self.y += self.vy
        
        # 边界处理
        if self.x < 0:
            self.x = 0
            self.vx = abs(self.vx) * 0.5
        if self.x > SW:
            self.x = SW
            self.vx = -abs(self.vx) * 0.5
        
        return reward
    
    def check_collision(self, terrain_points, landing_start, landing_end):
        """检测是否碰撞或着陆"""
        # 找到飞船下方的地形高度
        terrain_x = int(self.x)
        if terrain_x < 0 or terrain_x >= len(terrain_points):
            return False, 0
        
        # 线性插值找到精确的地形高度
        idx = terrain_x // 10
        if idx >= len(terrain_points) - 1:
            idx = len(terrain_points) - 2
        
        x1, y1 = terrain_points[idx]
        x2, y2 = terrain_points[idx + 1]
        t = (terrain_x - x1) / (x2 - x1)
        ground_y = y1 + t * (y2 - y1)
        
        # 飞船底部位置（飞船高度约30像素）
        ship_bottom = self.y + 15
        
        if ship_bottom >= ground_y:
            # 接触地面
            safe_speed = abs(self.vy) < 2.5
            safe_angle = abs(self.angle) < 25
            safe_horizontal = abs(self.vx) < 2
            on_landing_zone = landing_start <= self.x <= landing_end
            
            if safe_speed and safe_angle and safe_horizontal and on_landing_zone:
                self.landed = True
                return True, 1000  # 成功着陆奖励
            else:
                self.crashed = True
                return True, -1000  # 坠毁惩罚
        
        return False, 0


# ========= 绘制函数 =========
def draw_terrain(terrain_points):
    """绘制不平整的火星地形"""
    # 填充地形
    points_filled = [(0, SH)] + terrain_points + [(SW, SH)]
    pygame.draw.polygon(WINDOW, MARS_BROWN, points_filled)
    
    # 绘制地形轮廓
    pygame.draw.lines(WINDOW, MARS_LIGHT, False, terrain_points, 3)
    
    # 绘制着陆区域标记
    for point in terrain_points:
        if landing_start <= point[0] <= landing_end:
            pygame.draw.circle(WINDOW, GREEN, (int(point[0]), int(point[1]) + 5), 3)


def draw_lander(lander, action):
    """绘制飞船和喷射效果"""
    # 保存当前变换
    original_surface = pygame.Surface((40, 40), pygame.SRCALPHA)
    
    # 绘制飞船主体（三角形）
    ship_points = [
        (20, 5),   # 顶部
        (5, 35),   # 左下
        (35, 35),  # 右下
    ]
    pygame.draw.polygon(original_surface, WHITE, ship_points, 2)
    pygame.draw.polygon(original_surface, (200, 200, 200), ship_points)
    
    # 绘制窗户
    pygame.draw.circle(original_surface, BLUE, (20, 20), 5)
    
    # 旋转飞船
    rotated = pygame.transform.rotate(original_surface, -lander.angle)
    rect = rotated.get_rect(center=(lander.x, lander.y))
    WINDOW.blit(rotated, rect)
    
    # 绘制喷射火焰
    if action == 1:  # 主推进器
        flame_length = random.randint(15, 25)
        flame_points = [
            (lander.x - 8, lander.y + 15),
            (lander.x + 8, lander.y + 15),
            (lander.x, lander.y + 15 + flame_length),
        ]
        # 外层火焰（橙色）
        pygame.draw.polygon(WINDOW, ORANGE, flame_points)
        # 内层火焰（黄色）
        inner_flame = [
            (lander.x - 4, lander.y + 15),
            (lander.x + 4, lander.y + 15),
            (lander.x, lander.y + 15 + flame_length - 5),
        ]
        pygame.draw.polygon(WINDOW, YELLOW, inner_flame)
    
    if action == 2:  # 左转推进器（右侧喷射）
        flame_points = [
            (lander.x + 15, lander.y + 10),
            (lander.x + 25, lander.y + 10),
            (lander.x + 20, lander.y + 20),
        ]
        pygame.draw.polygon(WINDOW, ORANGE, flame_points)
        pygame.draw.polygon(WINDOW, YELLOW, [
            (lander.x + 17, lander.y + 10),
            (lander.x + 23, lander.y + 10),
            (lander.x + 20, lander.y + 17),
        ])
    
    if action == 3:  # 右转推进器（左侧喷射）
        flame_points = [
            (lander.x - 25, lander.y + 10),
            (lander.x - 15, lander.y + 10),
            (lander.x - 20, lander.y + 20),
        ]
        pygame.draw.polygon(WINDOW, ORANGE, flame_points)
        pygame.draw.polygon(WINDOW, YELLOW, [
            (lander.x - 23, lander.y + 10),
            (lander.x - 17, lander.y + 10),
            (lander.x - 20, lander.y + 17),
        ])


def draw_ui(generation, score, epsilon, lander):
    """绘制UI信息"""
    gen_text = Font.render(f"Gen: {generation}", True, WHITE)
    score_text = Font.render(f"Score: {score}", True, WHITE)
    epsilon_text = Font.render(f"Epsilon: {epsilon:.3f}", True, WHITE)
    fuel_text = Font.render(f"Fuel: {int(lander.fuel)}", True, WHITE)
    status_text = Font.render("TRAINING", True, YELLOW)
    
    WINDOW.blit(gen_text, (10, 10))
    WINDOW.blit(score_text, (10, 35))
    WINDOW.blit(epsilon_text, (10, 60))
    WINDOW.blit(fuel_text, (10, 85))
    WINDOW.blit(status_text, (SW - 120, 10))


# ========= 游戏主循环 =========
def game_loop():
    generation = 1
    scores = []
    epsilon = EPSILON_START
    
    # 生成地形
    global terrain_points, landing_start, landing_end
    terrain_points, landing_start, landing_end = generate_terrain()
    
    while True:
        lander = Lander()
        score = 0
        done = False
        
        while not done:
            for event in pygame.event.get():
                if event.type == QUIT:
                    # 绘制学习曲线
                    if scores:
                        plt.figure(figsize=(10, 6))
                        plt.plot(range(1, len(scores) + 1), scores, 'b-', linewidth=2)
                        plt.xlabel('Generation')
                        plt.ylabel('Score')
                        plt.title('Mars Lander AI Learning Curve')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.show()
                    pygame.quit()
                    sys.exit()
            
            # 获取状态
            state = lander.get_state()
            
            # Epsilon-greedy 策略选择动作
            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                action = np.argmax(Q[state])
            
            # 应用动作
            reward = lander.apply_action(action)
            
            # 检测碰撞
            collision, collision_reward = lander.check_collision(terrain_points, landing_start, landing_end)
            reward += collision_reward
            
            # 获取新状态
            new_state = lander.get_state()
            
            # Q-learning 更新
            best_next = np.max(Q[new_state])
            Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])
            
            # 绘制
            WINDOW.fill(BLACK)
            draw_terrain(terrain_points)
            draw_lander(lander, action)
            draw_ui(generation, score, epsilon, lander)
            
            # 绘制着陆区域标记线
            pygame.draw.line(WINDOW, GREEN, (landing_start, 0), (landing_start, SH), 2)
            pygame.draw.line(WINDOW, GREEN, (landing_end, 0), (landing_end, SH), 2)
            
            pygame.display.update()
            FPSCLOCK.tick(FPS)
            
            score += 1
            
            if collision:
                done = True
                scores.append(score)
                
                # 重新生成地形
                terrain_points, landing_start, landing_end = generate_terrain()
                
                # 衰减 epsilon
                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
                generation += 1
                
                print(f"Generation {generation-1}: Score={score}, Epsilon={epsilon:.3f}")


if __name__ == "__main__":
    print("Starting Mars Lander AI Training...")
    print("Watch the AI learn to land on uneven terrain!")
    game_loop()