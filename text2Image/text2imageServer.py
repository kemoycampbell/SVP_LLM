import base64
import threading
import queue
from io import BytesIO
from flask import Flask, request, jsonify
import torch
from diffusers import DiffusionPipeline
from flask import render_template


# ----------------------------
# SETTING UP THE MODEL configurations
# ----------------------------
# this is the model ID from Hugging Face that we will be using
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# by default we will use the cpu but if the GPU is available, we will switch to that instead
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# setup the pipeline
# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32
# ).to(device)

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to(device)


# turn off the safety checker so we dont run into issues with certain prompts
#pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))


# ----------------------------
# Flask + Queue Setup
# ----------------------------
app = Flask(__name__)
priority = queue.Queue()
tasks = {}  # the key will be the client id and the value will have a list of prompts for that client

# we will use this to store the result of the image generation
# This is a simple in memory storage type of thing to eliminate the need for a db
# we are storing the result in form of dictionary where the key is the client_id and the value is the base64 encoded image or error message
results_store = {}

processed_results = []


# ----------------------------
# Background worker to process the queue
# ----------------------------
def image_generation():
    while True:
        client_id = priority.get()  # blocking call, no busy wait

        if client_id not in results_store:
            results_store[client_id] = []

        try:
            if client_id in tasks and tasks[client_id]:
                prompt = tasks[client_id].pop(0)  # get first prompt

                # generate the image using the stable diffusion pipeline
                image = pipe(prompt).images[0]

                # convert the image to base64 string
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

                # store the result in the results store
                results_store[client_id].append(img_str)

                processed_results.append((client_id, prompt, img_str))

            

        except Exception as e:
            results_store[client_id].append(f"ERROR:{str(e)}")

        finally:
            priority.task_done()


threading.Thread(target=image_generation, daemon=True).start()


# ----------------------------
# API Endpoints
# ----------------------------
@app.route("/generate/image", methods=["POST"])
def generate_image():
    # Parse the request from the client
    data = request.get_json()
    # ensure that the prompt and client_id are provided
    if not data or "prompt" not in data or "client_id" not in data:
        return jsonify({"error": "Missing 'prompt' or 'client_id'"}), 400

    # Extract the prompt and client_id
    prompt = data["prompt"]
    client_id = data["client_id"]

    # Initialize tasks list for client if not exist
    if client_id not in tasks:
        tasks[client_id] = []

    # Append new prompt task
    tasks[client_id].append(prompt)

    # this client_id is next in line for processing
    priority.put(client_id)

    return jsonify({"status": "queued", "client_id": client_id, "prompt": prompt})


@app.route("/result/<client_id>", methods=["GET"])
def get_result(client_id):
    if client_id in results_store:
        results = results_store[client_id]
        if results:
            result = results.pop(0)
            if result.startswith("ERROR:"):
                return jsonify({"status": "error", "error": result.replace("ERROR:", "")})

            return jsonify({"status": "done", "client_id": client_id, "image_base64": result})

    return jsonify({"status": "no task pending"})


@app.route("/processed-tasks-json", methods=["GET"])
def processed_tasks_json():
    # Return the processed_results list as JSON
    return jsonify([
        {"client_id": task[0], "prompt": task[1], "img_str": task[2]}
        for task in reversed(processed_results)
    ])

@app.route("/tasks", methods=["GET"])
def processed_tasks_view():
    return render_template("tasks.html")


# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
