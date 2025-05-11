import pygame
import cv2
import numpy as np
from ultralytics import YOLO
import mss

# Initialize YOLO model
model = YOLO('C:/object-detection-realtime/best-new.pt')  # Replace with your custom model path

# Pygame setup
pygame.init()
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Object Detection on Screen")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Drone parameters
drone_pos = [screen_width // 2, screen_height - 50]
drone_speed = 5

def draw_drone(screen, pos):
    pygame.draw.circle(screen, RED, pos, 15)

# Object parameters
class Obstacle:
    def __init__(self, x, y, width, height, obj_name):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.obj_name = obj_name

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, (self.x, self.y, self.width, self.height))
        font = pygame.font.SysFont(None, 24)
        label = font.render(self.obj_name, True, BLACK)
        screen.blit(label, (self.x, self.y - 20))

# Map class IDs to object names
class_names = {0: "airplane", 1: "balloon", 2: "bird", 3: "helicopter", 4: "kite"}

def avoid_objects(detections, drone_pos):
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])  # Bounding box coordinates
        obj_center_x = (x1 + x2) // 2
        obj_center_y = (y1 + y2) // 2

        # Calculate distance from the drone to the object
        dist_x = obj_center_x - drone_pos[0]
        dist_y = obj_center_y - drone_pos[1]
        distance = np.sqrt(dist_x**2 + dist_y**2)

        if distance < 100:
            # Move drone away from the object
            if dist_x > 0:
                drone_pos[0] -= drone_speed
            else:
                drone_pos[0] += drone_speed

            if dist_y > 0:
                drone_pos[1] -= drone_speed
            else:
                drone_pos[1] += drone_speed

    # Keep drone within bounds
    drone_pos[0] = max(0, min(drone_pos[0], screen_width))
    drone_pos[1] = max(0, min(drone_pos[1], screen_height))

    return drone_pos

# Screen capture setup
sct = mss.mss()
monitor = sct.monitors[1]  # Change this if you have multiple monitors

# Main game loop
running = True
while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture the screen
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    # Process YOLO detections
    results = model(frame_rgb)

    detections = []
    if hasattr(results, 'pred') and results.pred is not None:
        for det in results.pred[0]:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            if conf > 0.5:
                obj_name = class_names.get(int(cls), "unknown")
                detections.append([x1, y1, x2, y2, conf, obj_name])

    # Create obstacles from detections
    obstacles = [Obstacle(int(x1), int(y1), int(x2 - x1), int(y2 - y1), obj_name) for x1, y1, x2, y2, conf, obj_name in detections]

    # Avoid detected objects
    drone_pos = avoid_objects(detections, drone_pos)

    # Draw detected objects
    for obstacle in obstacles:
        obstacle.draw(screen)

    # Draw drone
    draw_drone(screen, drone_pos)

    # Update display
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
