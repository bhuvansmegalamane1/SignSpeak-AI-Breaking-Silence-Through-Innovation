import cv2
import mediapipe as mp
import pyttsx3
import numpy as np
import time
from gesture_guide import GESTURE_GUIDE

class SignLanguageInterpreter:
    def __init__(self):
        # Initialize MediaPipe components
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Initialize guide variables
        self.show_guide = False
        self.guide_pages = list(GESTURE_GUIDE['Basic Hand Positions'].items())
        self.current_guide_page = 0
        
        # Initialize tracking variables
        self.last_sign = None
        self.last_spoken_time = time.time()
        self.speak_cooldown = 2.0  # seconds between speaking signs
        
        # Motion tracking
        self.prev_landmarks = None
        self.motion_threshold = 0.1
        self.gesture_history = []
        self.gesture_history_max = 30
        self.body_position_history = []
        
    def detect_full_body_motion(self, pose_landmarks):
        if pose_landmarks is None:
            return None
            
        # Get key body points
        shoulders = [pose_landmarks.landmark[11], pose_landmarks.landmark[12]]  # Left and right shoulders
        hands = [pose_landmarks.landmark[15], pose_landmarks.landmark[16]]      # Left and right wrists
        hips = [pose_landmarks.landmark[23], pose_landmarks.landmark[24]]       # Left and right hips
        head = pose_landmarks.landmark[0]                                       # Nose
        
        # Store current body position
        current_position = {
            'shoulders': [(s.x, s.y, s.z) for s in shoulders],
            'hands': [(h.x, h.y, h.z) for h in hands],
            'hips': [(h.x, h.y, h.z) for h in hips],
            'head': (head.x, head.y, head.z)
        }
        
        self.body_position_history.append(current_position)
        if len(self.body_position_history) > self.gesture_history_max:
            self.body_position_history.pop(0)
        
        return current_position
    
    def detect_facial_expression(self, face_landmarks):
        if face_landmarks is None:
            return None
            
        # Detect basic facial expressions based on landmark positions
        # This is a simplified version - you can expand based on your needs
        landmarks = face_landmarks.landmark
        
        # Example: detect smile by measuring mouth corners
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        top_mouth = landmarks[0]
        bottom_mouth = landmarks[17]
        
        mouth_ratio = abs(top_mouth.y - bottom_mouth.y) / abs(left_mouth.x - right_mouth.x)
        
        if mouth_ratio < 0.3:  # Threshold for smile detection
            return "smile"
        return None
    
    def recognize_complex_sign(self, hand_landmarks, pose_landmarks, face_landmarks):
        # Get hand features
        hand_features = self.get_hand_features(hand_landmarks) if hand_landmarks else None
        
        # Get body motion
        body_position = self.detect_full_body_motion(pose_landmarks)
        
        # Get facial expression
        expression = self.detect_facial_expression(face_landmarks)
        
        # Analyze the complete gesture
        if len(self.body_position_history) >= 2:
            prev_pos = self.body_position_history[-2]
            curr_pos = self.body_position_history[-1]
            
            # SLEEP - Head tilted on palm
            if hand_features and body_position:
                head_tilt = abs(curr_pos['head'][1] - prev_pos['head'][1])
                hand_near_head = any(
                    abs(h[1] - curr_pos['head'][1]) < 0.2 for h in curr_pos['hands']
                )
                if head_tilt > 0.1 and hand_near_head:
                    return "SLEEP"
            
            # HAPPY - Circular chest motion
            if len(self.body_position_history) >= 4:
                hand_path = [pos['hands'][0] for pos in self.body_position_history[-4:]]
                if self.detect_circular_motion(hand_path) and expression == "smile":
                    return "HAPPY"
            
            # SAD - Hands moving down from eyes
            hands_near_eyes = all(
                abs(h[1] - curr_pos['head'][1]) < 0.3 for h in prev_pos['hands']
            )
            hands_moving_down = all(
                curr_pos['hands'][i][1] > prev_pos['hands'][i][1] for i in range(2)
            )
            if hands_near_eyes and hands_moving_down:
                return "SAD"
            
            # EMERGENCY - Hands waving above head
            hands_above_head = all(
                h[1] < curr_pos['head'][1] - 0.2 for h in curr_pos['hands']
            )
            hands_moving = any(
                abs(curr_pos['hands'][i][1] - prev_pos['hands'][i][1]) > 0.1 for i in range(2)
            )
            if hands_above_head and hands_moving:
                return "EMERGENCY"
            
            # DOCTOR - Checking pulse motion
            right_hand_near_left_wrist = (
                abs(curr_pos['hands'][1][0] - curr_pos['hands'][0][0]) < 0.2 and
                abs(curr_pos['hands'][1][1] - curr_pos['hands'][0][1]) < 0.2
            )
            if right_hand_near_left_wrist:
                return "DOCTOR"
                
            # More complex signs...
            # Add recognition patterns for other signs that require body motion
            
        return None
    
    def recognize_sign(self, hand_landmarks):
        """Recognize basic hand signs based on finger positions."""
        features = self.get_hand_features(hand_landmarks)
        
        # Get finger tip positions relative to wrist
        finger_states = []
        for i in range(5):  # 5 fingers
            # Every finger has 7 features, extension is at index 6
            extension = features[i * 7 + 6]
            finger_states.append(extension > 0.1)  # True if finger is extended
        
        # Define basic signs based on finger states
        if all(finger_states):  # All fingers extended
            return "STOP"
        elif finger_states[1] and finger_states[2] and not any(finger_states[i] for i in [0,3,4]):
            return "PEACE"
        elif finger_states[1] and not any(finger_states[i] for i in [0,2,3,4]):
            return "YES"
        elif finger_states[4] and not any(finger_states[i] for i in [0,1,2,3]):
            return "NO"
        elif finger_states[0] and finger_states[1] and not any(finger_states[i] for i in [2,3,4]):
            return "HELLO"
        elif finger_states[1] and finger_states[2] and finger_states[3] and not any(finger_states[i] for i in [0,4]):
            return "THANK YOU"
        elif finger_states[1] and finger_states[2] and finger_states[3] and finger_states[4] and not finger_states[0]:
            return "HELP"
        elif finger_states[2] and finger_states[3] and finger_states[4] and not any(finger_states[i] for i in [0,1]):
            return "PLEASE"
            
        return None
        
    def get_hand_features(self, hand_landmarks):
        """Extract relevant features from hand landmarks for gesture recognition."""
        features = []
        
        # Get wrist position as reference point
        wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
        
        # Calculate relative positions of finger tips and bases relative to wrist
        finger_landmarks = [
            (mp.solutions.hands.HandLandmark.THUMB_TIP, mp.solutions.hands.HandLandmark.THUMB_CMC),
            (mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP),
            (mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP),
            (mp.solutions.hands.HandLandmark.RING_FINGER_TIP, mp.solutions.hands.HandLandmark.RING_FINGER_MCP),
            (mp.solutions.hands.HandLandmark.PINKY_TIP, mp.solutions.hands.HandLandmark.PINKY_MCP)
        ]
        
        for tip_id, base_id in finger_landmarks:
            tip = hand_landmarks.landmark[tip_id]
            base = hand_landmarks.landmark[base_id]
            
            # Calculate relative positions
            rel_tip_x = tip.x - wrist.x
            rel_tip_y = tip.y - wrist.y
            rel_tip_z = tip.z - wrist.z
            
            rel_base_x = base.x - wrist.x
            rel_base_y = base.y - wrist.y
            rel_base_z = base.z - wrist.z
            
            # Calculate finger extension (distance from base to tip)
            extension = np.sqrt(
                (tip.x - base.x)**2 +
                (tip.y - base.y)**2 +
                (tip.z - base.z)**2
            )
            
            # Add features for this finger
            features.extend([
                rel_tip_x, rel_tip_y, rel_tip_z,
                rel_base_x, rel_base_y, rel_base_z,
                extension
            ])
        
        # Add angles between adjacent fingers
        for i in range(4):
            tip1 = hand_landmarks.landmark[finger_landmarks[i][0]]
            tip2 = hand_landmarks.landmark[finger_landmarks[i+1][0]]
            
            # Calculate angle between fingers
            angle = np.arctan2(tip2.y - tip1.y, tip2.x - tip1.x)
            features.append(angle)
        
        return np.array(features)
    
    def detect_circular_motion(self, points):
        if len(points) < 4:
            return False
            
        # Calculate the center of the points
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        
        # Calculate angles from center to each point
        angles = []
        for p in points:
            angle = np.arctan2(p[1] - center_y, p[0] - center_x)
            angles.append(angle)
            
        # Check if angles form a circular pattern
        angle_diffs = [angles[i+1] - angles[i] for i in range(len(angles)-1)]
        return all(diff > 0 for diff in angle_diffs) or all(diff < 0 for diff in angle_diffs)
    
    def process_frame(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        hand_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        
        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Recognize sign if this is the primary (first detected) hand
                if hand_landmarks == hand_results.multi_hand_landmarks[0]:
                    sign = self.recognize_sign(hand_landmarks)
                    if sign:
                        cv2.putText(frame, f"Sign: {sign}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Speak the sign if it's new
                        if sign != self.last_sign:
                            self.last_sign = sign
                            self.speak(sign)
        
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        
        # Draw face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None)
        
        # Draw guide if enabled
        frame = self.draw_guide(frame)
        return frame
    
    def draw_guide(self, frame):
        if self.show_guide and self.guide_pages:
            sign, instruction = self.guide_pages[self.current_guide_page]
            
            # Get current category
            category = None
            for cat, signs in GESTURE_GUIDE.items():
                if sign in signs:
                    category = cat
                    break
            
            # Draw semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame.shape[0]-180), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            
            # Draw category and sign
            cv2.putText(frame, f"Category: {category}", (10, frame.shape[0]-160),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Sign: {sign}", (10, frame.shape[0]-130),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Split instruction into multiple lines if needed
            words = instruction.split()
            lines = []
            current_line = words[0]
            for word in words[1:]:
                if len(current_line + " " + word) < 40:
                    current_line += " " + word
                else:
                    lines.append(current_line)
                    current_line = word
            lines.append(current_line)
            
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (10, frame.shape[0]-90+i*30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
        
    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
    
    def run(self):
        cv2.namedWindow('Sign Language Interpreter', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Sign Language Interpreter', 1024, 768)
        cv2.moveWindow('Sign Language Interpreter', 300, 100)
        
        cap = cv2.VideoCapture(0)
        
        print("\nSign Language Interpreter - Controls:")
        print("- Press 'g' to show/hide gesture guide")
        print("- Press 'n' for next gesture in guide")
        print("- Press 'p' for previous gesture in guide")
        print("- Press 'q' to quit")
        print("\nAvailable Categories:")
        for category in GESTURE_GUIDE.keys():
            print(f"- {category}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            cv2.imshow('Sign Language Interpreter', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                self.show_guide = not self.show_guide
            elif key == ord('n'):
                self.current_guide_page = (self.current_guide_page + 1) % len(self.guide_pages)
            elif key == ord('p'):
                self.current_guide_page = (self.current_guide_page - 1) % len(self.guide_pages)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    interpreter = SignLanguageInterpreter()
    interpreter.run()
