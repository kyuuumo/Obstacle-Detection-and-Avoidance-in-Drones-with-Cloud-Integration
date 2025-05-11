import pygame
import cv2
import numpy as np
from ultralytics import YOLO
import random

model = YOLO('C:/object-detection-realtime/best1.pt')

pygame.init()
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("dodge")
clock = pygame.time.Clock()


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

drone_pos = [screen_width // 2, screen_height - 50]
drone_speed = 5

def draw_drone(screen, pos):
    pygame.draw.circle(screen, RED, pos, 15)


class Obstacle:
    def __init__(self, x, y, speed, obj_type):
        self.x = x
        self.y = y
        self.speed = speed
        self.obj_type = obj_type

    def move(self):
        self.y += self.speed
        if self.y > screen_height:
            self.y = -50
            self.x = random.randint(0, screen_width)

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, (self.x, self.y, 40, 40))

obstacles = [Obstacle(random.randint(0, screen_width), random.randint(-600, 0), random.randint(3, 6), "object") for _ in range(5)]

def avoid_objects(detections, drone_pos):
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        obj_center_x = (x1 + x2) // 2
        obj_center_y = (y1 + y2) // 2

        dist_x = obj_center_x - drone_pos[0]
        dist_y = obj_center_y - drone_pos[1]
        distance = np.sqrt(dist_x**2 + dist_y**2)

        if distance < 100:

            if dist_x > 0:
                drone_pos[0] -= drone_speed
            else:
                drone_pos[0] += drone_speed

            if dist_y > 0:
                drone_pos[1] -= drone_speed
            else:
                drone_pos[1] += drone_speed

    drone_pos[0] = max(0, min(drone_pos[0], screen_width))
    drone_pos[1] = max(0, min(drone_pos[1], screen_height))

    return drone_pos

running = True
while running:
    screen.fill(WHITE)


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    for obstacle in obstacles:
        obstacle.move()
        obstacle.draw(screen)


    frame = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255 
    results = model(frame)

    detections = []
    if hasattr(results, 'pred') and results.pred is not None:
        for det in results.pred[0]:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            if conf > 0.5:
                detections.append([x1, y1, x2, y2, conf, cls])

    drone_pos = avoid_objects(detections, drone_pos)


    draw_drone(screen, drone_pos)


    pygame.display.flip()
    clock.tick(30)

pygame.quit()
