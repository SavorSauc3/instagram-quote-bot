# Code to upload to instacalming channel
import torch
from ai_utils import DiffusionPipeline, TextGenerator
#from upload_utils import InstagramUploader
from wand.image import Image
from wand.drawing import Drawing
from wand.font import Font
import random
import csv
import json

# CLASS UTILS
# DIFFUSION PIPELINE
# TEXT GENERATOR




# Load LLM, load the csv file full of topics into a list
text_generator = TextGenerator()
csv_file_path = 'topics.csv'
topics = None
with open(csv_file_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    topics = next(reader)
# Access a random topic from the list to be used for the quote
random_topic = topics[random.randint(0, len(topics) - 1)].lstrip()
print(f"N-Topics: {len(topics)}")
print(f"Using topic: {random_topic}")
# Generate a quote with the topic that was randomly picked
prompt_text_zero = f"""
You are a quote generator, which generates one short quote based on the prompt I provide. Make sure to only generate one quote.
Generate a quote about {random_topic}.
"""
quote_text = text_generator.generate_text(prompt_text_zero, max_new_tokens=256)
print(f"Generated Quote: {quote_text}")
torch.cuda.empty_cache()
# Generate a prompt for the SDXL model to produce a background image
prompt_text_zero = f"""
You are generating 1 sentence describing an image that matches the following quote.
{quote_text}
"""
prompt_text = text_generator.generate_text(prompt_text_zero, max_new_tokens=70)
print(f"Using prompt: {prompt_text}")
torch.cuda.empty_cache()
# Run the prompt through the text-to-image model
text_to_image = DiffusionPipeline(model_base="stabilityai/stable-diffusion-xl-base-1.0", model_repo="ByteDance/SDXL-Lightning", model_ckpt="sdxl_lightning_8step_unet.safetensors")
text_to_image.generate_image(prompt_text, num_inference_steps=8, guidance_scale=0.1, output_path="image_output.png")
print("image was saved")

# Load the image, add the caption to the image, save as new image
with Image(filename="image_output.png") as canvas:
    with Drawing() as context:
        left, top, width, height = 45, 45, 820, 600
        context.fill_color = 'rgba(0, 0, 0, 0.5)'
        context.rectangle(left=left, top=top, width=width, height=height,radius=20)
        font = Font(r"c:\Windows\Fonts\BRADHITC.TTF",color='white', size=50)
        context(canvas)
        canvas.caption(quote_text, font=font, width=800, left=50, top=50)
    canvas.save(filename="instacalming_output.jpg")

caption = f"""
Today's topic: {random_topic}
-
-
-
-
-
-
-
-
-
-
-
#calming #calm #instagram #{random_topic} #inspirational #instacalming
"""
with open('personas.json') as f:
    data = json.load(f)
# Login to instagram and post the image with the caption
# instagram_credentials = data['personas'][0]['instagram']
# username = instagram_credentials['username']
# password = instagram_credentials['password']
# uploader = InstagramUploader(username=username, password=password)
# uploader.login()
# uploader.upload_post(file_path='instacalming_output.jpg', caption=caption, type='photo')