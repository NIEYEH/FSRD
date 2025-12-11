import json
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, LlavaForConditionalGeneration
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialRelationGenerator:
    def __init__(
        self,
        llava_model_path: str = "",
        llama3_model_path: str = "",
        data_dir: str = "",
        detection_dir: str = "",
        output_dir: str = ""
    ):
        self.data_dir = Path(data_dir)
        self.detection_dir = Path(detection_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Loading LLaVA model...")
        self.processor = AutoProcessor.from_pretrained(llava_model_path)
        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            llava_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        logger.info("Loading Llama3 model for synthesis...")
        self.llama3_tokenizer = AutoTokenizer.from_pretrained(llama3_model_path)
        # Set pad_token to avoid attention_mask warning
        if self.llama3_tokenizer.pad_token is None:
            self.llama3_tokenizer.pad_token = self.llama3_tokenizer.eos_token
            self.llama3_tokenizer.pad_token_id = self.llama3_tokenizer.eos_token_id
        
        self.llama3_model = AutoModelForCausalLM.from_pretrained(
            llama3_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.prompts = {
            "direction": """Based on the image and the following detected objects, describe the directional relationships between objects. Use these rules:
- Horizontal directions: left side, right side, directly left, directly right
- Vertical directions: above, below, directly above, directly below
- Diagonal: upper left, upper right, lower left, lower right
- Fuzzy modifiers: slightly, somewhat, approximately, roughly

Detected objects:
{objects}

Describe ONLY the directional relationships between these detected objects. Format: "Object B is [direction][modifier] of Object A". Do not describe anything else about the scene.""",
            
            "distance": """Based on the image and the following detected objects, provide DETAILED distance descriptions between objects. Use these distance terms and be specific:
- Very close distance: adjacent, touching, close to, nearly touching, right next to
- Near distance: nearby, beside, next to, a short distance away
- Medium distance: moderately separated, not far, relatively close, some distance apart
- Far distance: distant, far away, remote, well separated

Detected objects:
{objects}

For EACH pair of detected objects, describe:
1. Their approximate distance (use object size as reference, e.g., "approximately 1/4 of the boy's width apart", "about half the table's length away")
2. Whether they appear close, medium, or far apart
3. Specific distance indicators when possible

Be thorough and provide detailed distance information for all object pairs. Only describe distances between the detected objects listed above.""",
            
            "contact": """Based on the image and the following detected objects, describe the contact and containment relationships:
- Contact relationships: adjacent, separated, touching, not touching
- Containment relationships: inside, outside, contains, contained by

Detected objects:
{objects}

Describe ONLY whether these detected objects are touching each other or if there are containment relationships. Do not describe other aspects.""",
            
            "confidence": """Based on the image and the following detected objects with their confidence scores, evaluate the spatial relationship reliability:

Detected objects (with confidence scores):
{objects_with_scores}

Evaluate ONLY:
1. Which spatial relationships are most clear and reliable
2. Which relationships might be ambiguous

Keep evaluation concise and focused on the detected objects only."""
        }
        
        self.synthesis_prompt = """Synthesize the following spatial relationship descriptions into ONE natural, coherent paragraph.

Original spatial description:
{spatial_caption}

Directional relationships:
{direction_desc}

Distance relationships:
{distance_desc}

Contact and containment relationships:
{contact_desc}

Relationship reliability:
{confidence_desc}

CRITICAL REQUIREMENTS:
1. Create a SINGLE flowing paragraph that integrates all spatial information naturally
2. Preserve ONLY the scene elements and objects mentioned in the original spatial description
3. Focus heavily on DISTANCE information - this is the most important aspect
4. Include directional details to provide context
5. Mention contact/containment only when clearly relevant
6. Do NOT introduce new objects or elements not in the original description
7. Do NOT describe scene context, lighting, mood, or other aspects beyond spatial relationships
8. Do NOT repeat the same information multiple times
9. Keep the description concise and focused on spatial relationships only

Generate a natural, single-paragraph spatial description:"""

    def load_data(self, image_id: str):
        """Load image, spatial caption, and detections"""
        image_path = self.data_dir / f"{image_id}.jpg"
        json_path = self.data_dir / f"{image_id}.json"
        detection_path = self.detection_dir / f"{image_id}_detections.json"
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"JSON not found: {json_path}")
        if not detection_path.exists():
            raise FileNotFoundError(f"Detection not found: {detection_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        with open(json_path, 'r') as f:
            caption_data = json.load(f)
        
        with open(detection_path, 'r') as f:
            detections = json.load(f)
        
        if 'spatial_caption' not in caption_data:
            logger.warning(f"Missing 'spatial_caption' in {json_path}, using empty string")
            caption_data['spatial_caption'] = ""
        
        if 'coca_caption' not in caption_data:
            logger.warning(f"Missing 'original_caption' in {json_path}, using spatial_caption as fallback")
            caption_data['coca_caption'] = caption_data.get('spatial_caption', "")
        
        return image, caption_data, detections

    def format_objects(self, detections: List[Dict], include_scores: bool = False) -> str:
        lines = []
        for i, det in enumerate(detections):
            category = det['category']
            location = det['location']
            if include_scores:
                score = det['score']
                lines.append(f"{i+1}. {category} (confidence: {score:.2f}): location [{location[0]:.1f}, {location[1]:.1f}, {location[2]:.1f}, {location[3]:.1f}]")
            else:
                lines.append(f"{i+1}. {category}: location [{location[0]:.1f}, {location[1]:.1f}, {location[2]:.1f}, {location[3]:.1f}]")
        return "\n".join(lines)

    def generate_with_llava(self, image: Image.Image, prompt: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(self.llava_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.llava_model.generate(
                **inputs,
                max_new_tokens=600,  
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        if "ASSISTANT:" in generated_text:
            response = generated_text.split("ASSISTANT:")[-1].strip()
        elif "assistant" in generated_text.lower():
            response = generated_text.split("assistant")[-1].strip()
        else:
            # Remove the prompt from the output
            response = generated_text[len(prompt_text):].strip()
        
        return response

    def synthesize_descriptions(self, spatial_caption: str, descriptions: Dict[str, str]) -> str:
        synthesis_text = self.synthesis_prompt.format(
            spatial_caption=spatial_caption,
            direction_desc=descriptions['direction'],
            distance_desc=descriptions['distance'],
            contact_desc=descriptions['contact'],
            confidence_desc=descriptions['confidence']
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that synthesizes spatial descriptions into natural language."},
            {"role": "user", "content": synthesis_text}
        ]
        
        input_ids = self.llama3_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True  
        ).to(self.llama3_model.device)
        
        attention_mask = (input_ids != self.llama3_tokenizer.pad_token_id).long()
        
        with torch.no_grad():
            outputs = self.llama3_model.generate(
                input_ids,
                attention_mask=attention_mask, 
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.llama3_tokenizer.pad_token_id
            )
        
        final_desc = self.llama3_tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return final_desc.strip()

    def process_single_image(self, image_id: str):
        logger.info(f"Processing image {image_id}")
        
        try:
            image, caption_data, detections = self.load_data(image_id)
            spatial_caption = caption_data['spatial_caption']
            coca_caption = caption_data['coca_caption']
            
            if not detections:
                logger.warning(f"No detections found for image {image_id}, skipping")
                return None
            
            objects_str = self.format_objects(detections, include_scores=False)
            objects_with_scores_str = self.format_objects(detections, include_scores=True)
            
            descriptions = {}
            
            logger.info("  Generating direction relations...")
            descriptions['direction'] = self.generate_with_llava(
                image,
                self.prompts['direction'].format(objects=objects_str)
            )
            
            logger.info("  Generating distance relations...")
            descriptions['distance'] = self.generate_with_llava(
                image,
                self.prompts['distance'].format(objects=objects_str)
            )
            
            logger.info("  Generating contact/containment relations...")
            descriptions['contact'] = self.generate_with_llava(
                image,
                self.prompts['contact'].format(objects=objects_str)
            )
            
            logger.info("  Generating confidence assessment...")
            descriptions['confidence'] = self.generate_with_llava(
                image,
                self.prompts['confidence'].format(objects_with_scores=objects_with_scores_str)
            )
            
            logger.info("  Synthesizing final description...")
            expanded_caption = self.synthesize_descriptions(spatial_caption, descriptions)
            
            output = {
                "image_id": image_id,
                "coca_caption": coca_caption,
                "spatial_caption": spatial_caption,
                "expanded_caption": expanded_caption,
                "intermediate_descriptions": {
                    "direction": descriptions['direction'],
                    "distance": descriptions['distance'],
                    "contact": descriptions['contact'],
                    "confidence": descriptions['confidence']
                }
            }
            
            output_path = self.output_dir / f"{image_id}_expanded.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            logger.info(f"  Saved to {output_path}")
            return output
            
        except Exception as e:
            logger.error(f"Error processing {image_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def process_dataset(self):
        """Process entire dataset"""
        image_files = sorted(self.data_dir.glob("*.jpg"))
        
        success_count = 0
        error_count = 0
        
        for image_file in image_files:
            image_id = image_file.stem
            result = self.process_single_image(image_id)
            if result is not None:
                success_count += 1
            else:
                error_count += 1
        
        logger.info(f"\nProcessing complete: {success_count} succeeded, {error_count} failed")

def main():
    generator = SpatialRelationGenerator()
    
    generator.process_dataset()

if __name__ == "__main__":
    main()
