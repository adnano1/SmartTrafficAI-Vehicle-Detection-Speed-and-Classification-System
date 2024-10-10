import numpy as np
from time import time

previous_positions = {}
fps = 30  # Assumed FPS of video

def calculate_speed(current_positions):
    global previous_positions
    speeds = {}

    current_time = time()
    for id, pos in current_positions.items():
        if id in previous_positions:
            old_pos = previous_positions[id]
            speed = np.linalg.norm(np.array(pos) - np.array(old_pos)) * fps / (current_time - start_time)
            speeds[id] = speed

    previous_positions = current_positions.copy()
    return speeds
