from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import sys


class QwenVLActor:
    def __init__(self, model_path: str, temperature: float):
        local_files_only = model_path != "Qwen/Qwen2.5-VL-7B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", local_files_only=local_files_only
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.temperature = temperature

    def respond(self, prompt: str, obs: str = None) -> str:
        """
        Generate response based on text prompt and image.
        
        Args:
            prompt: Text prompt for the model
            obs: Path to the image file (for compatibility with custom model interface)
            
        Returns:
            Generated text response
        """
        # Use obs as image_path for compatibility
        image_path = obs
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1500,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
            )
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            return output_text.strip()
        except Exception as e:
            print(f"--= !!! Invalid QWEN Request: {e} !!! =--", file=sys.stderr)
            sys.exit(1)