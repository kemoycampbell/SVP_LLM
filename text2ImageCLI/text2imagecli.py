import requests
import uuid
import base64
from io import BytesIO
from PIL import Image
import threading
import time
import cv2
import numpy as np
from ollama import Client
import yaml
from rich import print
from rich.console import Console

console = Console()

# -------------------------
# Configuration
# -------------------------
base_url = "http://localhost:5000"
client_id = str(uuid.uuid4())
OLLAMA_SERVER_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:latest"

client = Client(host=OLLAMA_SERVER_URL)

# Load the LLM prompt template
with open("prompt.yaml", "r") as file:
    prompt_template = yaml.safe_load(file)

# -------------------------
# Functions
# -------------------------
def clean_prompt_with_LLM(user_prompt: str) -> str:
    messages = []
    for msg in prompt_template["clean_prompt_expert"]:
        content = msg["content"]
        if msg["role"] == "user":
            content = content.replace("{prompt}", user_prompt)
        messages.append({"role": msg["role"], "content": content})

    response = client.chat(model=LLM_MODEL, messages=messages, stream=False)
    return response.message.content.strip()


def generate_image(prompt):
    try:
        response = requests.post(
            f"{base_url}/generate/image",
            json={"prompt": prompt, "client_id": client_id},
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        return result.get("status", "unknown")
    except requests.RequestException as e:
        console.print(f"[red][Generate Error][/red]: {e}")
        return "error"


def display_image(image_base64):
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imshow("Generated Image", img_cv2)
    if cv2.waitKey(10000) == 27:  # ESC key to close early
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()


def poll_for_images(interval=3):
    while True:
        try:
            response = requests.get(f"{base_url}/result/{client_id}", timeout=5)
            response.raise_for_status()
            result = response.json()
            if "image_base64" in result:
                console.print("[green]New image received![/green]")
                display_image(result["image_base64"])
        except requests.RequestException as e:
            console.print(f"[red][Polling Error][/red]: {e}")
        time.sleep(interval)

# -------------------------
# Main program loop
# -------------------------
if __name__ == "__main__":
    thread = threading.Thread(target=poll_for_images, daemon=True)
    thread.start()

    while True:
        user_prompt = console.input("[cyan]Enter your prompt (or 'exit' to quit): [/cyan]")

        cleaned_prompt = clean_prompt_with_LLM(user_prompt)
        console.print(f"[yellow]Cleaned prompt:[/yellow] [white]{cleaned_prompt}[/white]")

        if cleaned_prompt.lower() in ["exit", "quit"]:
            console.print("[red]Exiting program.[/red]")
            break

        status = generate_image(cleaned_prompt)
        console.print(f"[magenta]Prompt sent, status:[/magenta] {status}")
