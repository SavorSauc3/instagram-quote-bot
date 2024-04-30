# Import Dependencies
import torch
from ctransformers import AutoModelForCausalLM, AutoConfig
from transformers import AutoTokenizer, pipeline
import soundfile as sf
from TTS.api import TTS as TTSAPI
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Generates text using a large language model
class TextGenerator:
    def __init__(self, model_name_or_path="TheBloke/Mistral-7b-instruct-v0.2-GGUF", tokenizer_path="mistralai/Mistral-7B-Instruct-v0.2", config_max_new_tokens=2000, config_context_length=4000, gpu_layers=50):
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.config.max_new_tokens = config_max_new_tokens
        config.config.context_length = config_context_length
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            model_file="mistral-7b-instruct-v0.2.Q8_0.gguf",
            model_type="mistral",
            gpu_layers=gpu_layers,
            hf=True,
            config=config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self.pipe = pipeline(model=self.model, tokenizer=self.tokenizer, task='text-generation')

    def generate_text(self, prompt_text, max_new_tokens=4000):
        messages = [{"role": "user", "content": prompt_text}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipe(prompt, max_new_tokens=max_new_tokens)
        generated_text = outputs[0]["generated_text"].split("[/INST]")[1]
        return generated_text
    
    def write_file(self, text, file_name):
        with open(file_name, "w") as file:
            file.write(text)
        print("Generated text has been written to ", file_name) # Writes file to current directory of code being used

# Creates music from text prompts
class TextToAudioSynthesizer:
    def __init__(self, model_name="facebook/musicgen-stereo-large", device="cuda:0", torch_dtype=torch.float16):
        self.synthesiser = pipeline("text-to-audio", model_name, device=device, torch_dtype=torch_dtype)

    def generate_audio(self, text, max_new_tokens=1500, output_file="musicgen_output.wav"):
        """
        Don't use a value for max_new_tokens higher than 1500
        More than 1500 tokens produces gibberish
        """
        music =  self.synthesiser(text, forward_params={"max_new_tokens": max_new_tokens})
        sf.write(output_file, music["audio"][0].T, music["sampling_rate"])

# Generates speech using text, and a speaker voice input to mimic
class TextToSpeech:
    def __init__(self, model_path):
        """
        For best current model path, try using "tts_models/multilingual/multi-dataset/xtts_v2"
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTSAPI(model_path).to(self.device)

    def list_models(self):
        return self.tts.list_models()

    def generate_speech(self, text, speaker_wav, language="en", file_path="tts_output.wav"):
        self.tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=file_path)

# Generates images from text
class DiffusionPipeline:
    def __init__(self, model_base, model_repo, model_ckpt):
        """
        If unsure where to start, use the following:
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!
        """
        self.unet = UNet2DConditionModel.from_config(model_base, subfolder="unet").to("cuda", torch.float16)
        self.unet.load_state_dict(load_file(hf_hub_download(model_repo, model_ckpt), device="cuda"))
        self.pipe = StableDiffusionXLPipeline.from_pretrained(model_base, unet=self.unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")

    def generate_image(self, text, num_inference_steps, guidance_scale, output_path):
        self.pipe(text, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0].save(output_path)

