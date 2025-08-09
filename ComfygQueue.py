import re
import comfy
import random
import torch
import nodes
import os
import numpy as np
from PIL import Image
import folder_paths

class ComfygQueue:
    CATEGORY = "Workflow/Batch"

    @classmethod  
    def INPUT_TYPES(cls):
        return {
            "required": {
                "res_presets": ("STRING", {
                    "default": "1024x1024,1152x896,896x1152",
                    "multiline": False
                }),
                "save_prefix": ("STRING", {"default": "batch_res_"}),
                "batch_seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                # These will be passed through from connected nodes
                "model": ("MODEL",),
                "positive": ("CONDITIONING",), 
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),  # If you want to use an existing latent instead of empty
                "vae": ("VAE",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_info",)
    FUNCTION = "process"
    OUTPUT_NODE = True

    def save_image_tensor(self, image_tensor, filename):
        """Save a ComfyUI image tensor to file"""
        try:
            # ComfyUI images are in format [batch, height, width, channels] with values 0-1
            if image_tensor.dim() == 4:
                image_tensor = image_tensor[0]  # Take first image from batch
            
            # Convert to numpy and scale to 0-255
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            
            # Convert to PIL and save
            image_pil = Image.fromarray(image_np)
            
            # Get ComfyUI's output directory
            output_dir = folder_paths.get_output_directory()
            filepath = os.path.join(output_dir, filename)
            
            image_pil.save(filepath, "PNG")
            return filepath
            
        except Exception as e:
            print(f"Error saving image {filename}: {e}")
            return None

    def process(self, res_presets, save_prefix, batch_seed, **kwargs):
        # Parse resolutions
        res_list = []
        for res_str in res_presets.split(","):
            match = re.match(r"\s*(\d+)x(\d+)", res_str.strip())
            if match:
                res_list.append((int(match.group(1)), int(match.group(2))))
        
        if not res_list:
            return ("No valid resolutions found",)
        
        # Get the optional parameters with defaults
        steps = kwargs.get('steps', 20)
        cfg = kwargs.get('cfg', 7.0)
        sampler_name = kwargs.get('sampler_name', 'euler')
        scheduler = kwargs.get('scheduler', 'normal') 
        denoise = kwargs.get('denoise', 1.0)
        
        # Check if we have the required nodes
        model = kwargs.get('model')
        positive = kwargs.get('positive')
        negative = kwargs.get('negative')
        vae = kwargs.get('vae')
        
        if not all([model, positive, negative, vae]):
            return ("Missing required connections: model, positive, negative, vae",)
        
        # Initialize ComfyUI nodes
        empty_latent = nodes.EmptyLatentImage()
        ksampler = nodes.KSampler()
        decode = nodes.VAEDecode()
        
        images = []
        info_list = []
        saved_files = []
        
        for i, (w, h) in enumerate(res_list):
            seed = batch_seed if batch_seed != -1 else random.randint(0, 2**31-1)
            
            try:
                # Create latent
                if 'latent' in kwargs and kwargs['latent'] is not None:
                    latent_image = empty_latent.generate(w, h, 1)[0]
                else:
                    latent_image = empty_latent.generate(w, h, 1)[0]
                
                # Sample
                samples = ksampler.sample(model, seed, steps, cfg, sampler_name, scheduler,
                                        positive, negative, latent_image, denoise)[0]
                
                # Decode
                image = decode.decode(vae, samples)[0]
                images.append(image)
                
                # Save each image with original resolution
                filename = f"{save_prefix}{w}x{h}_seed{seed}.png"
                saved_path = self.save_image_tensor(image, filename)
                
                if saved_path:
                    saved_files.append(saved_path)
                    info_list.append(f"✓ {w}x{h} (seed: {seed}) → {filename}")
                else:
                    info_list.append(f"✗ {w}x{h} (seed: {seed}) → SAVE FAILED")
                
            except Exception as e:
                info_list.append(f"✗ {w}x{h} - ERROR: {str(e)}")
                print(f"Error at {w}x{h}: {e}")
        
        # Create summary
        summary = f"Generated and saved {len(saved_files)} images:\n" + "\n".join(info_list)
        
        if saved_files:
            summary += f"\n\nFiles saved to: {folder_paths.get_output_directory()}"
        
        return (summary,)


# Queue Trigger Node - Simulates multiple "Queue Prompt" clicks
class ComfygQueueTrigger:
    CATEGORY = "Workflow/Batch"

    @classmethod  
    def INPUT_TYPES(cls):
        return {
            "required": {
                "res_presets": ("STRING", {
                    "default": "1024x1024,1152x896,896x1152",
                    "multiline": False
                }),
                "batch_seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "execution_mode": (["queue_multiple", "return_data_only"], {"default": "queue_multiple"}),
                "modify_seeds": ("BOOLEAN", {"default": True}),  # Whether to change seeds for each resolution
            },
            "hidden": {
                # This gets the current workflow from ComfyUI
                "prompt": "PROMPT"
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "queue_data")
    FUNCTION = "trigger_batch"
    OUTPUT_NODE = True

    def parse_resolutions(self, res_presets):
        """Parse resolution string into list of (width, height) tuples"""
        res_list = []
        for res_str in res_presets.split(","):
            match = re.match(r"\s*(\d+)x(\d+)", res_str.strip())
            if match:
                res_list.append((int(match.group(1)), int(match.group(2))))
        return res_list

    def find_nodes_in_workflow(self, workflow):
        """Find KSampler and EmptyLatentImage nodes in the workflow"""
        ksampler_nodes = []
        empty_latent_nodes = []
        
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
                
            class_type = node_data.get("class_type", "")
            
            if class_type == "KSampler":
                ksampler_nodes.append(node_id)
            elif class_type == "EmptyLatentImage":
                empty_latent_nodes.append(node_id)
        
        return ksampler_nodes, empty_latent_nodes

    def modify_workflow_for_resolution(self, workflow, width, height, seed, ksampler_nodes, empty_latent_nodes):
        """Modify workflow to use specific resolution and seed"""
        modified_workflow = copy.deepcopy(workflow)
        
        # Update EmptyLatentImage nodes with new resolution
        for node_id in empty_latent_nodes:
            if node_id in modified_workflow:
                node = modified_workflow[node_id]
                if "inputs" in node:
                    node["inputs"]["width"] = width
                    node["inputs"]["height"] = height
        
        # Update KSampler nodes with new seed
        for node_id in ksampler_nodes:
            if node_id in modified_workflow:
                node = modified_workflow[node_id]
                if "inputs" in node:
                    node["inputs"]["seed"] = seed
        
        return modified_workflow

    def queue_workflow(self, workflow):
        """Submit a workflow to ComfyUI's queue"""
        try:
            import server
            from execution import PromptExecutor
            
            # Generate unique client ID
            client_id = str(uuid.uuid4())
            
            # Create prompt in the format ComfyUI expects
            prompt_data = {
                "prompt": workflow,
                "client_id": client_id
            }
            
            # Get the prompt executor (this is how ComfyUI queues internally)
            prompt_executor = PromptExecutor(server.PromptServer.instance)
            
            # Queue the prompt (this is essentially what happens when you click "Queue Prompt")
            valid = prompt_executor.validate_prompt(workflow)
            if valid[0]:
                # Add to queue
                prompt_id = server.PromptServer.instance.prompt_queue.put(workflow, client_id)
                return True, prompt_id
            else:
                return False, f"Invalid workflow: {valid[1]}"
                
        except Exception as e:
            return False, f"Queue error: {str(e)}"

    def trigger_batch(self, res_presets, batch_seed, execution_mode, modify_seeds, prompt):
        """Main function that triggers multiple queue executions"""
        
        # Get the current workflow
        if prompt is None:
            return ("ERROR: No workflow data available", "")
        
        # Handle both workflow formats
        workflow = prompt.get('prompt', prompt) if isinstance(prompt, dict) else prompt
        
        if not workflow:
            return ("ERROR: Empty workflow", "")
        
        # Parse resolutions
        resolutions = self.parse_resolutions(res_presets)
        if not resolutions:
            return ("ERROR: No valid resolutions found", "")
        
        # Find relevant nodes in workflow
        ksampler_nodes, empty_latent_nodes = self.find_nodes_in_workflow(workflow)
        
        if not ksampler_nodes:
            return ("ERROR: No KSampler nodes found in workflow", "")
        
        if not empty_latent_nodes:
            return ("ERROR: No EmptyLatentImage nodes found in workflow", "")
        
        # Prepare batch data
        batch_jobs = []
        status_messages = []
        
        for i, (width, height) in enumerate(resolutions):
            # Generate seed for this job
            if modify_seeds:
                seed = batch_seed if batch_seed != -1 else random.randint(0, 2**31-1)
            else:
                seed = batch_seed if batch_seed != -1 else -1  # Let KSampler handle random seed
            
            # Create modified workflow for this resolution/seed
            modified_workflow = self.modify_workflow_for_resolution(
                workflow, width, height, seed, ksampler_nodes, empty_latent_nodes
            )
            
            job_info = {
                "job_id": i + 1,
                "resolution": f"{width}x{height}",
                "seed": seed,
                "ksampler_nodes": ksampler_nodes,
                "empty_latent_nodes": empty_latent_nodes
            }
            batch_jobs.append(job_info)
            
            if execution_mode == "queue_multiple":
                # Actually queue the workflow
                success, result = self.queue_workflow(modified_workflow)
                
                if success:
                    status_messages.append(f"✓ Queued {width}x{height} (seed: {seed}) - Prompt ID: {result}")
                else:
                    status_messages.append(f"✗ Failed to queue {width}x{height}: {result}")
            else:
                # Just prepare the data
                status_messages.append(f"📋 Prepared {width}x{height} (seed: {seed})")
        
        # Create summary
        if execution_mode == "queue_multiple":
            summary = f"Queued {len(batch_jobs)} workflows:\n" + "\n".join(status_messages)
            summary += f"\n\nCheck the queue panel - {len(batch_jobs)} jobs should be processing!"
        else:
            summary = f"Prepared {len(batch_jobs)} workflow configurations:\n" + "\n".join(status_messages)
        
        # Return batch data as JSON
        batch_data = json.dumps(batch_jobs, indent=2)
        
        return (summary, batch_data)

NODE_CLASS_MAPPINGS = {
    "ComfygQueue": ComfygQueue,
    "ComfygQueueTrigger": ComfygQueueTrigger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfygQueue": "Batch Resolution Generator",
    "ComfygQueueTrigger": "Multi-Queue Trigger"
}