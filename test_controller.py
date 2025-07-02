# test_controller.py
import cv2
from controller import ControllerAgent
from agents.vision_agent import VisionAgent
from agents.recognition_agent import RecognitionAgent
from adk import Agent, Workflow
from agents.vision_agent import VisionAgent
from agents.recognition_agent import RecognitionAgent


controller = ControllerAgent()
frame = cv2.imread('path_to_hand_image.jpg')
controller.process_frame(frame)


# controller.py

# Define your workflow
workflow = Workflow(
    agents=[VisionAgent(), RecognitionAgent()],
    # ... define orchestration logic ...
)

if __name__ == "__main__":
    workflow.run()