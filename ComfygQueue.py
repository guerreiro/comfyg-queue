import re
import comfy
import random

class ComfygQueue:
    CATEGORY = "Workflow/Batch"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {}),
                "neg_prompt": ("STRING", {"default": ""}),
                "res_presets": ("STRING", {
                    "default": (
                        "1024x1024,1152x896,896x1152,1216x832,832x1216,"
                        "1344x768,768x1344,1536x640,640x1536,1536x1024,"
                        "1824x1248,1024x1536,1248x1824,1536x1536"
                    )
                }),
                "manual_resolution": ("STRING", {
                    "default": "",
                    "optional": True
                }),
                "use_batch": ("BOOL", {"default": True}),
                "model": ("MODEL", {}),
                "sampler": ("SAMPLER", {}),
                "steps": ("INT", {"default": 20}),
                "cfg_scale": ("FLOAT", {"default": 5.0}),
                "denoise": ("FLOAT", {"default": 1.0}),
                "seed": ("INT", {"default": -1})
            }
        }
    RETURN_TYPES = ("IMAGE_LIST", "DICT_LIST")
    FUNCTION = "process"
    OUTPUT_NODE = True

    def process(self, prompt, neg_prompt, res_presets,
                manual_resolution, use_batch,
                model, sampler, steps, cfg_scale, denoise, seed):
        presets = []
        if use_batch or not manual_resolution:
            presets = [tuple(map(int, m.groups()))
                       for m in (re.match(r"\s*(\d+)x(\d+)", part)
                                 for part in res_presets.split(","))
                       if m]
        else:
            m = re.match(r"\s*(\d+)x(\d+)", manual_resolution)
            presets = [(int(m.group(1)), int(m.group(2)))] if m else []

        images = []
        infos = []
        for w, h in presets:
            cur_seed = random.randrange(0, 2**32-1) if seed < 0 else seed
            job = comfy.TextToImage(
                prompt=prompt,
                negative_prompt=neg_prompt,
                model=model,
                width=w, height=h,
                sampler_name=sampler,
                steps=steps,
                cfg=cfg_scale,
                denoise=denoise,
                seed=cur_seed
            )
            result = job.run()
            images.append(result.image)
            info = {"width": w, "height": h, "seed": cur_seed,
                    "steps": steps, "cfg": cfg_scale, "denoise": denoise}
            if neg_prompt:
                info["neg_prompt"] = neg_prompt
            infos.append(info)

        return (images, infos)

NODE_CLASS_MAPPINGS = {"ComfygQueue": ComfygQueue}