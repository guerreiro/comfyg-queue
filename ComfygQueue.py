import re
import os
import random
import json
import uuid
import copy
import time
import numpy as np
from PIL import Image, PngImagePlugin
import server
from server import PromptServer
import comfy
import torch
import nodes
import folder_paths

# ------------------------------
# Helper Functions
# ------------------------------
def parse_resolutions(res_string):
    """Convert '1024x1024,1152x896' → [(1024,1024),(1152,896)]"""
    return [
        (int(m.group(1)), int(m.group(2)))
        for r in res_string.split(",")
        if (m := re.match(r"\s*(\d+)x(\d+)", r.strip()))
    ]

def save_tensor_image(image_tensor, filename, output_subdir=""):
    """Save ComfyUI tensor image to PNG"""
    try:
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        if image_np.shape[-1] == 3:
            image_pil = Image.fromarray(image_np, "RGB")
        elif image_np.shape[-1] == 4:
            image_pil = Image.fromarray(image_np, "RGBA")
        else:
            image_pil = Image.fromarray(image_np[:, :, 0], "L")

        output_dir = folder_paths.get_output_directory()
        if output_subdir:
            output_dir = os.path.join(output_dir, output_subdir)
            os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)
        image_pil.save(filepath, "PNG")
        return filepath
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return None


# ------------------------------
# Batch Resolution Generator
# ------------------------------
class ComfygQueue:
    CATEGORY = "Workflow/Batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "res_presets": ("STRING", {"default": "1024x1024,1152x896,896x1152"}),
                "save_prefix": ("STRING", {"default": "ComfygQueue_Gen_"}),
                "batch_seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "incremental_seed": ("BOOLEAN", {"default": False}),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),  # optional
                "vae": ("VAE",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "output_subdir": ("STRING", {"default": ""}),
            },
            "hidden": {
                "prompt": "PROMPT",  # automatically provided by ComfyUI
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("save_info","file_list")
    FUNCTION = "process"
    OUTPUT_NODE = True

    def _normalize_extra_pnginfo(self, extra):
        # Some nodes/framework versions hand this in as [dict]; make it robust.
        if extra is None:
            return {}
        if isinstance(extra, list):
            for item in extra:
                if isinstance(item, dict):
                    return item
            return {}
        if isinstance(extra, dict):
            return extra
        return {}

    def save_image_with_metadata(self, image_tensor, filename, prompt, extra_pnginfo, seed, output_subdir=""):
        try:
            if image_tensor.dim() == 4:
                image_tensor = image_tensor[0]
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

            if image_np.shape[-1] == 3:
                image_pil = Image.fromarray(image_np, "RGB")
            elif image_np.shape[-1] == 4:
                image_pil = Image.fromarray(image_np, "RGBA")
            else:
                image_pil = Image.fromarray(image_np[:, :, 0], "L")

            output_dir = folder_paths.get_output_directory()
            if output_subdir:
                output_dir = os.path.join(output_dir, output_subdir)
                os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)

            # Build metadata exactly like stock SaveImage does.
            pnginfo = PngImagePlugin.PngInfo()
            try:
                if prompt is not None:
                    pnginfo.add_text("prompt", json.dumps(prompt))
                ep = self._normalize_extra_pnginfo(extra_pnginfo)
                if ep:
                    for k, v in ep.items():
                        # Stock SaveImage json.dumps every value.
                        pnginfo.add_text(k, json.dumps(v))
                
                pnginfo.add_text("seed", str(seed))
            except Exception as e:
                print(f"Warning: Could not embed metadata: {e}")

            image_pil.save(filepath, "PNG", pnginfo=pnginfo)
            return filepath
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            return None

    def process(self, res_presets, save_prefix, batch_seed, output_subdir="", prompt=None, extra_pnginfo=None, **kwargs):
        resolutions = parse_resolutions(res_presets)
        if not resolutions:
            return ("No valid resolutions found",)

        missing = [n for n in ["model", "positive", "negative", "vae"] if kwargs.get(n) is None]
        if missing:
            return (f"Missing required connections: {', '.join(missing)}",)

        steps, cfg = kwargs.get("steps", 20), kwargs.get("cfg", 7.0)
        sampler_name, scheduler = kwargs.get("sampler_name", "euler"), kwargs.get("scheduler", "normal")
        denoise = kwargs.get("denoise", 1.0)
        model, positive, negative, vae = kwargs["model"], kwargs["positive"], kwargs["negative"], kwargs["vae"]

        empty_latent = nodes.EmptyLatentImage()
        ksampler = nodes.KSampler()
        decode = nodes.VAEDecode()

        # Ensure we always have a dict for workflow
        if isinstance(prompt, dict) and "prompt" in prompt:
            workflow_dict = prompt["prompt"]
        elif isinstance(prompt, dict):
            workflow_dict = prompt
        else:
            workflow_dict = {}

        saved_files, info_list = [], []

        # Generate the base seed once for all resolutions
        if batch_seed == -1:
            base_seed = random.randint(0, 2**31 - 1)

        # Loop over resolutions but keep the same seed
        for w, h in resolutions:

            # Increment +1 for each resolution
            if kwargs.get("incremental_seed", False):
                base_seed += 1
            
            seed = base_seed
            try:
                # Latent optional
                if kwargs.get("latent") is not None:
                    latent_image = kwargs["latent"]
                else:
                    latent_image = empty_latent.generate(w, h, 1)[0]

                samples = ksampler.sample(model, seed, steps, cfg, sampler_name, scheduler,
                                          positive, negative, latent_image, denoise)[0]
                image = decode.decode(vae, samples)[0]

                filename = f"{save_prefix}_{w}x{h}_seed_{seed}.png"
                saved_path = self.save_image_with_metadata(
                    image, filename, prompt, extra_pnginfo, seed, output_subdir
                )

                if saved_path:
                    saved_files.append(saved_path)
                    info_list.append(f"✓ {w}x{h} (seed {seed}) → {filename}")
                else:
                    info_list.append(f"✗ {w}x{h} (seed {seed}) → SAVE FAILED")
            except Exception as e:
                info_list.append(f"✗ {w}x{h} ERROR: {e}")
                print(f"Error at {w}x{h}: {e}")

        summary = f"Generated {len(saved_files)} images with seed {base_seed}:\n" + "\n".join(info_list)
        if saved_files:
            summary += f"\n\nFiles saved to: {folder_paths.get_output_directory()}"
        return (summary, json.dumps(saved_files))


# ------------------------------
# Multi-Queue Trigger
# ------------------------------
class ComfygQueueTrigger:
    CATEGORY = "Workflow/Batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "res_presets": ("STRING", {"default": "1024x1024,1152x896,896x1152"}),
                "batch_seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "execution_mode": (["queue_multiple", "return_data_only"], {"default": "queue_multiple"}),
                "modify_seeds": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT"
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "queue_data")
    FUNCTION = "trigger_batch"
    OUTPUT_NODE = True

    def find_nodes(self, workflow):
        ksampler_nodes, empty_latent_nodes = [], []
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            if node_data.get("class_type") == "KSampler":
                ksampler_nodes.append(node_id)
            elif node_data.get("class_type") == "EmptyLatentImage":
                empty_latent_nodes.append(node_id)
        return ksampler_nodes, empty_latent_nodes

    def modify_workflow(self, workflow, width, height, seed, ksampler_nodes, empty_latent_nodes):
        for node_id in empty_latent_nodes:
            workflow[node_id]["inputs"]["width"] = width
            workflow[node_id]["inputs"]["height"] = height
        for node_id in ksampler_nodes:
            workflow[node_id]["inputs"]["seed"] = seed
        return workflow

    def queue_workflow(self, workflow):
        try:
            import server
            from execution import PromptExecutor
            client_id = str(uuid.uuid4())
            prompt_executor = PromptExecutor(server.PromptServer.instance)
            valid = prompt_executor.validate_prompt(workflow)
            if valid[0]:
                prompt_id = server.PromptServer.instance.prompt_queue.put(workflow, client_id)
                return True, prompt_id
            return False, f"Invalid workflow: {valid[1]}"
        except Exception as e:
            return False, f"Queue error: {e}"

    def trigger_batch(self, res_presets, batch_seed, execution_mode, modify_seeds, prompt):
        if not prompt:
            return ("ERROR: No workflow data available", "")

        workflow = prompt["prompt"] if isinstance(prompt, dict) and "prompt" in prompt else prompt
        resolutions = parse_resolutions(res_presets)
        if not resolutions:
            return ("ERROR: No valid resolutions found", "")

        ksampler_nodes, empty_latent_nodes = self.find_nodes(workflow)
        if not ksampler_nodes or not empty_latent_nodes:
            return ("ERROR: Missing KSampler or EmptyLatentImage nodes", "")

        batch_jobs, status_messages = [], []
        for i, (w, h) in enumerate(resolutions):
            seed = (batch_seed if batch_seed != -1 else random.randint(0, 2**31 - 1))
            if modify_seeds and batch_seed != -1:
                seed = batch_seed + i

            modified_workflow = self.modify_workflow(copy.deepcopy(workflow), w, h, seed,
                                                     ksampler_nodes, empty_latent_nodes)

            job_info = {"job_id": i + 1, "resolution": f"{w}x{h}", "seed": seed}
            batch_jobs.append(job_info)

            if execution_mode == "queue_multiple":
                success, result = self.queue_workflow(modified_workflow)
                if success:
                    status_messages.append(f"✓ Queued {w}x{h} (seed {seed}) - ID: {result}")
                else:
                    status_messages.append(f"✗ Failed {w}x{h}: {result}")
                time.sleep(0.05)  # avoid choking ComfyUI
            else:
                status_messages.append(f"📋 Prepared {w}x{h} (seed {seed})")

        summary = (f"Queued {len(batch_jobs)} workflows:\n" if execution_mode == "queue_multiple"
                   else f"Prepared {len(batch_jobs)} workflows:\n") + "\n".join(status_messages)

        return (summary, json.dumps(batch_jobs, indent=2))


# ------------------------------
# Node Mappings
# ------------------------------
NODE_CLASS_MAPPINGS = {
    "ComfygQueue": ComfygQueue,
    "ComfygQueueTrigger": ComfygQueueTrigger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfygQueue": "Batch Resolution Generator",
    "ComfygQueueTrigger": "Multi-Queue Trigger"
}
