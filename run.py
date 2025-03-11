import gradio
from src.yolo import detectObjects

demo = gradio.Interface(fn=detectObjects, inputs=gradio.Image(type="pil"), outputs="image")
    
if __name__ == "__main__":
    demo.launch(share=True)
