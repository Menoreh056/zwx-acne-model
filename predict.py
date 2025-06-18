import replicate
from cog import BasePredictor, Input, Path
import torch
from diffusers import StableDiffusionPipeline

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = StableDiffusionPipeline.from_single_file(
            "zwx_model.ckpt",
            torch_dtype=torch.float16
        ).to("cuda")

    def predict(self,
        prompt: str = Input(description="Prompt to generate image"),
    ) -> Path:
        image = self.pipe(prompt).images[0]
        image.save("output.png")
        return Path("output.png")
