# agents/animation_agent.py
from fastapi import FastAPI, UploadFile, File, Form
from controller import ControllerAgent

app = FastAPI()
controller = ControllerAgent()

@app.post("/submit_input/")
async def submit_input(text: str = Form(None), audio: UploadFile = File(None)):
    if audio:
        # Convert audio to text (implement this)
        text = speech_to_text(await audio.read())
    if not text:
        return {"error": "No input provided"}
    # Process text through your pipeline
    video_path = controller.process_text(text)  # You need to implement this
    return {"video_url": f"/get_video/{video_path}"}

@app.get("/get_video/{video_path}")
async def get_video(video_path: str):
    # Serve the video file
    return FileResponse(f"videos/{video_path}")