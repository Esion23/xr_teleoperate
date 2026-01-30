import numpy as np

class MockTeleData:
    def __init__(self):
        # Initialize with Identity matrices (4x4)
        self.left_wrist_pose = np.eye(4)
        self.right_wrist_pose = np.eye(4)
        self.head_pose = np.eye(4)
        
        # Initialize other fields usually provided by TeleVuer
        self.left_pinch = False
        self.right_pinch = False
        self.left_hand_confidence = 1.0
        self.right_hand_confidence = 1.0

        self.left_wrist_pose[:3, 3] = [0.3, 0.2, 0.3]
        self.right_wrist_pose[:3, 3] = [0.3, -0.2, 0.3]
        self.left_hand_joint = np.zeros(75) 
        self.right_hand_joint = np.zeros(75)
        
        self.left_hand_pos = np.zeros(75)
        self.right_hand_pos = np.zeros(75)
