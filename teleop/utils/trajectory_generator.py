import numpy as np
import time

class SquareTrajectoryGenerator:
    def __init__(self, center_pos, side_length=0.2, period=4.0):
        """
        Generate a square trajectory in the Y-Z plane (robot's frontal plane).
        
        Args:
            center_pos (np.array): [x, y, z] center of the square in Robot Arm Frame
            side_length (float): Length of the square side in meters
            period (float): Time to complete one full cycle in seconds
        """
        self.center_pos = np.array(center_pos)
        self.side_length = side_length
        self.period = period
        self.start_time = None
        
    def get_target(self, current_time):
        if self.start_time is None:
            self.start_time = current_time
            
        t = (current_time - self.start_time) % self.period
        phase = t / self.period # 0 to 1
        
        half = self.side_length / 2
        x = self.center_pos[0]
        cy, cz = self.center_pos[1], self.center_pos[2]
        
        if phase < 0.25:
            local_p = phase * 4
            y = (cy + half) + local_p * ((cy - half) - (cy + half))
            z = cz + half
            
        elif phase < 0.5:
            local_p = (phase - 0.25) * 4
            y = cy - half
            z = (cz + half) + local_p * ((cz - half) - (cz + half))
            
        elif phase < 0.75:
            local_p = (phase - 0.5) * 4
            y = (cy - half) + local_p * ((cy + half) - (cy - half))
            z = cz - half
            
        else:
            local_p = (phase - 0.75) * 4
            y = cy + half
            z = (cz - half) + local_p * ((cz + half) - (cz - half))
            
        return np.array([x, y, z])
