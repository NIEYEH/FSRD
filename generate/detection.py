import os
import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import spacy
from tqdm import tqdm
from spacy.tokens import Doc
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re

from transformers import (
    pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    GroundingDinoProcessor, GroundingDinoForObjectDetection,
    AutoProcessor, AutoModelForZeroShotObjectDetection,
    OwlViTProcessor, OwlViTForObjectDetection,
    AutoModelForCausalLM as Florence2Model
)

from torchvision.ops import box_convert
import torchvision.transforms as T_vision

from ensemble_boxes import weighted_boxes_fusion

CONFIG = {
    "DATA_DIR": "",
    
    "GROUNDING_DINO_MODEL": "",
    "OWLVIT_MODEL": "",
    "FLORENCE2_MODEL": "",
    
    "ZERO_SHOT_BERT_PATH": "",
    
    "MISTRAL_MODEL_PATH": "",
    
    "SPACY_MODEL": "en_core_web_trf",
    "OUTPUT_DIR": "",
    "VISUALIZATION_DIR": "",
    
    "GROUNDING_DINO_THRESHOLD": 0.36,
    "OWLVIT_THRESHOLD": 0.36,
    "FLORENCE2_THRESHOLD": 0.36,
    
    "WBF_IOU_THRESHOLD": 0.5,
    "WBF_SKIP_BOX_THRESHOLD": 0.0001,
    "WBF_CONF_TYPE": "avg",  # 'avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg'
    
    "USE_MISTRAL": True,  
}

BLOCKLIST_ROOTS = {
    'left', 'right', 'top', 'bottom', 'side', 'front', 'back', 'center', 'middle',
    'background', 'foreground', 'room', 'area', 'space', 'location', 'position',
    'part', 'edge', 'corner', 'view', 'example', 'end', 'stack', 
    'hand', 'head', 'face', 'foot', 'leg', 'arm', 'body', 'eye',
    'standing', 'oscillating'
}

class ObjectExtractor:
    def __init__(self, model_path, device="cuda", batch_size=8):
        self.device = device
        self.batch_size = batch_size
        
        self.nlp = spacy.load(CONFIG["SPACY_MODEL"])
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quantization_config,
            dtype=torch.float16
        )
        
        self.exclusion_words = self._load_exclusion_words()
        
        self.extraction_cache = {}
    
    def _load_exclusion_words(self):
        position_words = set(['left', 'right', 'top', 'bottom', 'front', 'back', 
                             'side', 'corner', 'center', 'middle', 'edge'])
        abstract_words = set(['time', 'idea', 'thought', 'view', 'moment', 'position',
                             'location', 'area', 'space', 'direction'])
        body_parts = set(['hand', 'foot', 'head', 'arm', 'leg', 'eye', 'ear', 'face',
                         'finger', 'toe', 'hair', 'mouth', 'nose'])
        prepositions = set(['in', 'on', 'at', 'by', 'with', 'from', 'to', 'for',
                           'of', 'about', 'against', 'between', 'into', 'through',
                           'during', 'before', 'after', 'above', 'below', 'up', 'down'])
        descriptors = set(['being', 'standing', 'sitting', 'lying', 'positioned',
                          'located', 'placed', 'present', 'visible', 'seen'])
        return position_words.union(abstract_words).union(body_parts).union(prepositions).union(descriptors)

    def preprocess_text(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'m", " am", text)
        return text

    def extract_candidate_objects(self, text):
        doc = self.nlp(text)
        candidates = []
        
        for chunk in doc.noun_chunks:
            phrase = chunk.text
            if (chunk.root.lemma_ not in self.exclusion_words and
                not (len(chunk) == 1 and chunk.root.pos_ == "PRON")):
                candidates.append({
                    "phrase": phrase,
                    "root": chunk.root.text,
                    "pos": chunk.root.pos_,
                    "deps": [token.dep_ for token in chunk]
                })
        
        compound_patterns = []
        for token in doc:
            if token.text == "of" and token.head.pos_ == "NOUN" and token.head.dep_ != "pobj":
                head = token.head
                for child in token.children:
                    if child.dep_ == "pobj" and child.pos_ == "NOUN":
                        start = min(head.i, child.i)
                        end = max(head.i, child.i)
                        phrase = doc[start:end+1].text
                        if all(word.lower() not in self.exclusion_words 
                               for word in [head.text.lower(), child.text.lower()]):
                            compound_patterns.append({
                                "phrase": phrase,
                                "root": head.text,
                                "pos": "COMPOUND",
                                "deps": ["compound"]
                            })
        
        candidates.extend(compound_patterns)
        return candidates

    def rule_based_filter(self, candidates):
        filtered = []
        for item in candidates:
            phrase = item["phrase"].lower()
            words = phrase.split()
            
            if all(word in self.exclusion_words for word in words):
                continue
            if re.match(r'^(in front of|next to|beside|on top of|behind)$', phrase):
                continue
            if words[0] in ['in', 'on', 'at', 'by'] and len(words) <= 2:
                continue
            if len(words) <= 3 and "being" in words and any(prep in words for prep in ['on', 'in', 'at']):
                continue
            filtered.append(item)
        
        return filtered

    def generate_llm_prompt(self, candidates):
        if not candidates:
            return ""
        objects_text = "\n".join([f"{i+1}. {item['phrase']}" for i, item in enumerate(candidates)])
        prompt = f"""<s>[INST] You are an object detection assistant. Your task is to identify real physical objects from text descriptions.

Please review this list of potential objects:
{objects_text}

For each numbered item, respond with ONLY "yes" or "no":
- Answer "yes" if the item is a concrete physical object that could be visually detected in an image
- Answer "no" if it's a location, direction, abstract concept, action or purely descriptive term

Examples:
- "table" → yes (physical furniture)
- "refrigerator" → yes (physical appliance)
- "stack of books" → yes (physical objects grouped together)
- "cell phone" → yes (physical device)
- "left side" → no (just a location)
- "standing" → no (just an action/position)
- "front" → no (just a direction)

Respond with numbered answers like:
1. yes
2. no
etc.
[/INST]</s>"""
        return prompt

    def verify_with_llm(self, candidates_batch):
        if not candidates_batch:
            return []
        prompt = self.generate_llm_prompt(candidates_batch)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "[/INST]" in full_response:
            response = full_response.split("[/INST]")[1].strip()
        else:
            prompt_end = prompt.replace("<s>[INST]", "").replace("[/INST]</s>", "").strip()
            if prompt_end in full_response:
                response = full_response.split(prompt_end)[-1].strip()
            else:
                response = full_response[-500:].strip()
        
        yes_indices = []
        pattern = r'(\d+)[\.:\)]?\s*(?:yes|Yes|YES)' 
        matches = re.finditer(pattern, response)
        
        for match in matches:
            try:
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(candidates_batch):
                    yes_indices.append(idx)
            except ValueError:
                continue
        
        if not yes_indices:
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if i < len(candidates_batch) and re.search(r'yes', line.lower()):
                    yes_indices.append(i)
        
        results = [candidates_batch[i]["phrase"] for i in yes_indices]
        return results

    def process_batches(self, all_candidates):
        verified_objects = []
        for i in range(0, len(all_candidates), self.batch_size):
            batch = all_candidates[i:i+self.batch_size]
            batch_results = self.verify_with_llm(batch)
            verified_objects.extend(batch_results)
        return verified_objects
    
    def normalize_object_name(self, phrase):
        doc = self.nlp(phrase)
        nouns = [token for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        
        if not nouns:
            return phrase.lower()
        
        core_noun = nouns[-1].text.lower()
        
        if " of " in phrase:
            match = re.search(r'(\w+)\s+of\s+(\w+)', phrase.lower())
            if match:
                common_compounds = [
                    "stack of", "pile of", "group of", "bunch of",
                    "set of", "pair of", "collection of", "bowl of",
                    "plate of", "cup of", "glass of", "bottle of"
                ]
                for compound in common_compounds:
                    if phrase.lower().startswith(compound):
                        return phrase.lower()
        
        return core_noun

    def extract_objects(self, text):
        text_hash = hash(text)
        if text_hash in self.extraction_cache:
            return self.extraction_cache[text_hash]
        
        processed_text = self.preprocess_text(text)
        candidates = self.extract_candidate_objects(processed_text)
        filtered_candidates = self.rule_based_filter(candidates)
        verified_objects = self.process_batches(filtered_candidates)
        
        normalized_objects = []
        seen_core_objects = set()
        
        for obj in verified_objects:
            core_obj = self.normalize_object_name(obj)
            if core_obj not in seen_core_objects:
                seen_core_objects.add(core_obj)
                normalized_objects.append(core_obj)
        
        self.extraction_cache[text_hash] = normalized_objects
        return normalized_objects

def deduplicate_keywords(keywords: List[str], nlp: spacy.language.Language) -> List[str]:
    root_to_longest_phrase: Dict[str, str] = {}
    
    docs = list(nlp.pipe(keywords, batch_size=50))
    
    for phrase, doc in zip(keywords, docs):
        if len(doc) == 0:
            continue
        root_lemma = doc[-1].lemma_.lower()
        if root_lemma not in root_to_longest_phrase:
            root_to_longest_phrase[root_lemma] = phrase
        else:
            existing_phrase = root_to_longest_phrase[root_lemma]
            if len(phrase) > len(existing_phrase):
                root_to_longest_phrase[root_lemma] = phrase
                
    final_list = list(root_to_longest_phrase.values())
    return final_list

def get_keywords_3_stage(text_data: dict, nlp: spacy.language.Language, extractor) -> List[str]:
    full_text = text_data.get("spatial_caption", "")
    if not full_text.strip():
        print("[Error] spatial_caption is empty.")
        return []

    if CONFIG["USE_MISTRAL"] and isinstance(extractor, ObjectExtractor):
        print("[Extraction] Using the Mistral object extractor...")
        objects = extractor.extract_objects(full_text)
        print(f"    [Extraction] [Result] Found {len(objects)} objects.")
        return objects
    else:
        print("    [Stage 1] Extracting noun phrases using spaCy...")
        doc = nlp(full_text)
        stage1_candidates = doc.noun_chunks
        
        stage2_candidates = []
        print("    [Stage 2] Rule-based filtering...")
        VALID_NOUN_TAGS = {'NN', 'NNS', 'NNP', 'NNPS'}
        
        for chunk in stage1_candidates:
            if len(chunk) == 0:
                continue
            root_pos = chunk.root.pos_
            root_lemma = chunk.root.lemma_.lower()
            last_token = chunk[-1] 
            last_token_tag = last_token.tag_
            
            if root_pos not in ('NOUN', 'PROPN'):
                continue
            if last_token_tag not in VALID_NOUN_TAGS:
                continue
            if root_lemma in BLOCKLIST_ROOTS:
                continue
            stage2_candidates.append(chunk.text)

        if not stage2_candidates:
            print("    [Stage 2] [Result] All candidate keywords were filtered out.")
            return []
        
        stage2_candidates = list(set([k.lower() for k in stage2_candidates]))
        
        print(f"    [Stage 3] AI semantic filtering ( {len(stage2_candidates)} candidate keywords)...")
        final_keywords_list = []
        classification_labels = ["detectable item or person", "abstract concept"]
        
        try:
            results = extractor(stage2_candidates, classification_labels, multi_label=False)
        except Exception as e:
            print(f"    [Stage 3] [Error] LLM classification failed: {e}")
            return [] 

        for res in results:
            if res['labels'][0] == "detectable item or person" and res['scores'][0] > 0.5:
                final_keywords_list.append(res['sequence'])
                
        if not final_keywords_list:
            print("    [Stage 3] [Result] All candidate keywords were filtered out.")
            return []

        final_keywords_list = list(set([k.lower() for k in final_keywords_list]))
        print(f"[Stage 3] [Result] {len(final_keywords_list)} keywords remaining.")
        return final_keywords_list

class MultiModelDetector:
    def __init__(self, device="cuda"):
        self.device = device
        self.models = {}
        self.processors = {}
        
        self.detection_cache = {}
        
    def load_grounding_dino(self, model_path):
        print(f"Loading GroundingDINO: {model_path}")
        processor = GroundingDinoProcessor.from_pretrained(model_path)
        model = GroundingDinoForObjectDetection.from_pretrained(model_path).to(self.device)
        model.eval()  
        self.models['grounding_dino'] = model
        self.processors['grounding_dino'] = processor
        print("✓ GroundingDINO loaded successfully.")
        
    def load_owlvit(self, model_path):
        print(f"Loading OWL-ViT: {model_path}")
        processor = OwlViTProcessor.from_pretrained(model_path)
        model = OwlViTForObjectDetection.from_pretrained(model_path).to(self.device)
        model.eval() 
        self.models['owlvit'] = model
        self.processors['owlvit'] = processor
        print("✓ OWL-ViT loaded successfully.")
        
    def load_florence2(self, model_path):
        print(f"Loading Florence-2: {model_path}")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = Florence2Model.from_pretrained(
            model_path, 
            trust_remote_code=True,
            attn_implementation="eager"  
        ).to(self.device)
        model.eval()  
        self.models['florence2'] = model
        self.processors['florence2'] = processor
        print("✓ Florence-2 loaded successfully.")
    
    def detect_grounding_dino(self, image: Image.Image, text_prompt: str, threshold: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        processor = self.processors['grounding_dino']
        model = self.models['grounding_dino']
        
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):  
            outputs = model(**inputs)
        
        w, h = image.size
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, target_sizes=[(h, w)]
        )
        
        result = results[0]
        mask = result["scores"] > threshold
        boxes = result["boxes"][mask].cpu().numpy()
        scores = result["scores"][mask].cpu().numpy()
        labels = [result["labels"][i] for i in range(len(result["labels"])) if mask[i]]
        
        return boxes, scores, labels
    
    def detect_owlvit(self, image: Image.Image, text_queries: List[str], threshold: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        processor = self.processors['owlvit']
        model = self.models['owlvit']
        
        inputs = processor(text=text_queries, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):  
            outputs = model(**inputs)
        
        w, h = image.size
        target_sizes = torch.tensor([[h, w]]).to(self.device)
        results = processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )
        
        boxes_list, scores_list, labels_list = [], [], []
        for i, (box, score, label) in enumerate(zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"])):
            if score > threshold:
                boxes_list.append(box.cpu().numpy())
                scores_list.append(score.cpu().item())
                labels_list.append(text_queries[label])
        
        boxes = np.array(boxes_list) if boxes_list else np.empty((0, 4))
        scores = np.array(scores_list) if scores_list else np.empty(0)
        
        return boxes, scores, labels_list
    
    def detect_florence2(self, image: Image.Image, text_prompt: str, threshold: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Florence-2 Detection"""
        processor = self.processors['florence2']
        model = self.models['florence2']
        
        task_prompt = "<OD>"
        
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            inputs = processor(text=task_prompt, images=image, return_tensors="pt")
            
            if "pixel_values" not in inputs:
                print("Florence-2 processor did not return pixel_values")
                return np.empty((0, 4)), np.empty(0), []
            
            if inputs["pixel_values"] is None:
                print("Florence-2 pixel_values is None")
                return np.empty((0, 4)), np.empty(0), []
            
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
        except Exception as e:
            print(f"Florence-2 input processing failed: {e}")
            return np.empty((0, 4)), np.empty(0), []
        
        try:
            with torch.no_grad(), torch.amp.autocast('cuda'): 
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512,  
                    num_beams=1,  
                    do_sample=False,
                    use_cache=False  
                )
        except Exception as e:
            print(f"Florence-2 generation failed: {e}")
            try:
                print(f"Trying again with simplified parameters...")
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=256,
                        use_cache=False
                    )
            except Exception as e2:
                print(f"Florence-2 retry failed: {e2}")
                return np.empty((0, 4)), np.empty(0), []
        
        try:
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
        except Exception as e:
            print(f"Florence-2 post-processing failed: {e}")
            return np.empty((0, 4)), np.empty(0), []
        
        boxes_list, scores_list, labels_list = [], [], []
        
        if parsed_answer is None:
            print("Florence-2 returned None")
            return np.empty((0, 4)), np.empty(0), []
        
        if not isinstance(parsed_answer, dict) or '<OD>' not in parsed_answer:
            print(f"Florence-2 returned unexpected format: {type(parsed_answer)}")
            return np.empty((0, 4)), np.empty(0), []
        
        od_result = parsed_answer['<OD>']
        if not isinstance(od_result, dict):
            print(f"Florence-2 OD result returned unexpected format: {type(od_result)}")
            return np.empty((0, 4)), np.empty(0), []
        
        if 'bboxes' not in od_result or 'labels' not in od_result:
            print("Florence-2 did not detect any objects")
            return np.empty((0, 4)), np.empty(0), []
        
        bboxes = od_result['bboxes']
        labels = od_result['labels']
        
        if not bboxes or not labels:
            print("Florence-2 detection results are empty")
            return np.empty((0, 4)), np.empty(0), []
        
        target_keywords = set([kw.strip().lower() for kw in text_prompt.split('.') if kw.strip()])
        
        for bbox, label in zip(bboxes, labels):
            label_lower = label.lower()
            if any(kw in label_lower or label_lower in kw for kw in target_keywords):
                boxes_list.append(bbox)
                scores_list.append(threshold + 0.1) 
                labels_list.append(label)
        
        boxes = np.array(boxes_list) if boxes_list else np.empty((0, 4))
        scores = np.array(scores_list) if scores_list else np.empty(0)
        
        return boxes, scores, labels_list

def normalize_boxes(boxes: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    normalized = boxes.copy()
    normalized[:, [0, 2]] /= img_width
    normalized[:, [1, 3]] /= img_height
    return normalized

def denormalize_boxes(boxes: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    denormalized = boxes.copy()
    denormalized[:, [0, 2]] *= img_width
    denormalized[:, [1, 3]] *= img_height
    return denormalized

def apply_wbf(
    all_boxes: List[np.ndarray],
    all_scores: List[np.ndarray],
    all_labels: List[List[str]],
    img_width: int,
    img_height: int,
    iou_thr: float = 0.5,
    skip_box_thr: float = 0.0001,
    conf_type: str = 'avg'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:

    normalized_boxes = [normalize_boxes(boxes, img_width, img_height) for boxes in all_boxes]
    
    all_unique_labels = list(set(label for labels in all_labels for label in labels))
    label_to_idx = {label: idx for idx, label in enumerate(all_unique_labels)}
    
    numeric_labels = []
    for labels in all_labels:
        numeric_labels.append([label_to_idx[label] for label in labels])
    
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        normalized_boxes,
        all_scores,
        numeric_labels,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        conf_type=conf_type
    )
    
    fused_boxes = denormalize_boxes(fused_boxes, img_width, img_height)
    
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    fused_label_names = [idx_to_label[int(idx)] for idx in fused_labels]
    
    return fused_boxes, fused_scores, fused_label_names

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def extract_core_noun(label: str, nlp) -> str:
    doc = nlp(label.lower())
    
    nouns = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    
    if nouns:
        return nouns[-1]
    
    return label.lower().replace(" ", "")

def filter_duplicate_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: List[str],
    nlp,
    iou_threshold: float = 0.9,
    allowed_keywords: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if len(boxes) == 0:
        return boxes, scores, labels
    
    if allowed_keywords is not None:
        allowed_core_nouns = set([extract_core_noun(kw, nlp) for kw in allowed_keywords])
        
        filtered_indices = []
        for i, label in enumerate(labels):
            core_noun = extract_core_noun(label, nlp)
            if core_noun in allowed_core_nouns:
                filtered_indices.append(i)
            else:
                print(f"Filtering out mismatched detections: '{label}' (core noun: '{core_noun}') is not present in the prompt.")
        
        if not filtered_indices:
            print("All detections have been filtered out")
            return np.empty((0, 4)), np.empty(0), []
        
        boxes = boxes[filtered_indices]
        scores = scores[filtered_indices]
        labels = [labels[i] for i in filtered_indices]
        print(f"After keyword filtering, {len(boxes)} detections remain.")
    
    n = len(boxes)
    keep_mask = np.ones(n, dtype=bool)
    
    core_nouns = [extract_core_noun(label, nlp) for label in labels]
    
    for i in range(n):
        if not keep_mask[i]:
            continue
            
        for j in range(i + 1, n):
            if not keep_mask[j]:
                continue
            
            if core_nouns[i] == core_nouns[j]:
                iou = calculate_iou(boxes[i], boxes[j])
                
                if iou > iou_threshold:
                    if scores[i] >= scores[j]:
                        keep_mask[j] = False
                        print(f"Removing duplicate detection: '{labels[j]}' (IoU={iou:.3f}, keeping '{labels[i]}')")
                    else:
                        keep_mask[i] = False
                        print(f"Removing duplicate detection: '{labels[i]}' (IoU={iou:.3f}, keeping '{labels[j]}')")
                        break
    
    filtered_boxes = boxes[keep_mask]
    filtered_scores = scores[keep_mask]
    filtered_labels = [labels[i] for i in range(n) if keep_mask[i]]
    
    removed_count = n - len(filtered_boxes)
    if removed_count > 0:
        print(f"Duplicate filtering completed, removed {removed_count} duplicate detections.")
    
    return filtered_boxes, filtered_scores, filtered_labels

_standardize_label_cache = {}

def standardize_label(label: str, nlp) -> str:
    if label in _standardize_label_cache:
        return _standardize_label_cache[label]
    
    doc = nlp(label.lower())
    
    nouns = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    
    if nouns:
        last_noun = nouns[-1]
        if last_noun in ['teddy', 'bear']:
            result = 'bear'
        else:
            result = last_noun
    else:
        result = label.lower().replace(" ", "_")
    
    _standardize_label_cache[label] = result
    return result

def process_image(
    detector: MultiModelDetector,
    nlp,
    extractor_or_classifier,
    image_path: str,
    json_path: str,
    output_dir: str,
    visualization_dir: str,
    device: str
):
    
    print(f"\n{'='*60}")
    print(f" > Image: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: Unable to read or parse JSON file {json_path}: {e}")
        return

    stage3_keywords = get_keywords_3_stage(data, nlp, extractor_or_classifier)
    if not stage3_keywords:
        print(f"Warning: Failed to extract keywords, skipping {json_path}")
        return
        
    deduplicated_keywords = deduplicate_keywords(stage3_keywords, nlp)
    if not deduplicated_keywords:
        print(f"Warning: Keywords empty after deduplication, skipping {json_path}")
        return
        
    text_prompt = " . ".join(deduplicated_keywords)
    print(f"    > Final prompt: {text_prompt}")
    try:
        pil_image = Image.open(image_path).convert("RGB")
        image_for_drawing = pil_image.copy()
        img_width, img_height = pil_image.size
    except Exception as e:
        print(f"Error: Unable to open image file {image_path}: {e}")
        return
    
    all_boxes, all_scores, all_labels = [], [], []
    
    print("\n[Detection Stage]")
    
    # 1. GroundingDINO
    try:
        print("GroundingDINO detecting...")
        boxes, scores, labels = detector.detect_grounding_dino(
            pil_image, text_prompt, CONFIG["GROUNDING_DINO_THRESHOLD"]
        )
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        print(f"Detected {len(boxes)} objects")
    except Exception as e:
        print(f"GroundingDINO detection failed: {e}")
        all_boxes.append(np.empty((0, 4)))
        all_scores.append(np.empty(0))
        all_labels.append([])
    
    # 2. OWL-ViT
    try:
        print("OWL-ViT detecting...")
        boxes, scores, labels = detector.detect_owlvit(
            pil_image, deduplicated_keywords, CONFIG["OWLVIT_THRESHOLD"]
        )
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        print(f"Detected {len(boxes)} objects")
    except Exception as e:
        print(f"OWL-ViT detection failed: {e}")
        all_boxes.append(np.empty((0, 4)))
        all_scores.append(np.empty(0))
        all_labels.append([])
    
    # 3. Florence-2
    try:
        print("Florence-2 detecting...")
        boxes, scores, labels = detector.detect_florence2(
            pil_image, text_prompt, CONFIG["FLORENCE2_THRESHOLD"]
        )
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        print(f"Detected {len(boxes)} objects")
    except Exception as e:
        print(f"Florence-2 detection failed: {e}")
        all_boxes.append(np.empty((0, 4)))
        all_scores.append(np.empty(0))
        all_labels.append([])
    
    print("\n[Pre-standardization Stage]")
    original_all_labels = []  
    standardized_all_labels = []
    
    all_unique_labels = set()
    for labels in all_labels:
        all_unique_labels.update(labels)
    
    label_mapping = {}
    if all_unique_labels:
        for label in all_unique_labels:
            label_mapping[label] = standardize_label(label, nlp)
    
    for model_idx, labels in enumerate(all_labels):
        original_labels = list(labels)
        standardized_labels = [label_mapping[label] for label in labels]
        
        original_all_labels.append(original_labels)
        standardized_all_labels.append(standardized_labels)
        
        unique_mappings = {}
        for orig, std in zip(original_labels, standardized_labels):
            if orig != std and orig not in unique_mappings:
                unique_mappings[orig] = std
        
        if unique_mappings:
            model_names = ['GroundingDINO', 'OWL-ViT', 'Florence-2']
            print(f"  [{model_names[model_idx]}] Standardized {len(unique_mappings)} labels:")
            for orig, std in unique_mappings.items():
                print(f"     {orig} → {std}")
    
    all_labels = standardized_all_labels
    
    print("\n[WBF Fusion Stage]")
    try:
        fused_boxes, fused_scores, fused_labels = apply_wbf(
            all_boxes, all_scores, all_labels,
            img_width, img_height,
            iou_thr=CONFIG["WBF_IOU_THRESHOLD"],
            skip_box_thr=CONFIG["WBF_SKIP_BOX_THRESHOLD"],
            conf_type=CONFIG["WBF_CONF_TYPE"]
        )
        print(f"WBF fusion completed, detected {len(fused_boxes)} objects")
    except Exception as e:
        print(f"WBF fusion failed: {e}")
        return
    
    print("\n[Duplicate Detection Filtering Stage]")
    if len(fused_boxes) > 0:
        fused_boxes, fused_scores, fused_labels = filter_duplicate_detections(
            fused_boxes,
            fused_scores,
            fused_labels,
            nlp,
            iou_threshold=0.85,  
            allowed_keywords=deduplicated_keywords
        )
        print(f"Filtered down to {len(fused_boxes)} detections")
    
    print("\n[Instance ID Assignment Stage]")
    category_counts = {}
    instance_ids = []
    
    for i, label in enumerate(fused_labels):
        if label not in category_counts:
            category_counts[label] = 0
        category_counts[label] += 1
        instance_id = f"{label}_{category_counts[label]}"
        instance_ids.append(instance_id)
    
    print(f"  ✓ Assigned {len(instance_ids)} instance IDs")
    if len(category_counts) < len(fused_labels):
        print(f"     Found {len(category_counts)} categories, {len(fused_labels)} instances")
        for cat, count in category_counts.items():
            if count > 1:
                print(f"       - {cat}: {count} instances")
    
    detections = []
    if len(fused_scores) > 0:
        for i in range(len(fused_scores)):
            detections.append({
                "instance_id": instance_ids[i],
                "category": fused_labels[i],
                "original_label": fused_labels[i],  
                "location": fused_boxes[i].tolist(),
                "score": float(fused_scores[i]),
            })
    
    if len(detections) > 0:
        draw = ImageDraw.Draw(image_for_drawing)
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (0, 255, 255), (255, 0, 255), (255, 128, 0), (128, 0, 255)
        ]
        
        try:
            font = ImageFont.load_default(size=20)
        except:
            try:
                font = ImageFont.load_default()
            except:
                class DummyFont:
                    def getbbox(self, text): return (0,0,10,10)
                font = DummyFont()

        for i, detection in enumerate(detections):
            box = detection["location"]
            label = detection["category"]
            score = detection["score"]
            color = colors[i % len(colors)]
            
            draw.rectangle(box, outline=color, width=3)
            text = f"{label}: {score:.2f}"
            
            try:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                text_width, text_height = 10, 10

            text_bg_origin = (box[0], box[1] - text_height - 4)
            text_bg_end = (box[0] + text_width + 4, box[1])
            
            draw.rectangle([text_bg_origin, text_bg_end], fill=color)
            draw.text((box[0] + 2, box[1] - text_height - 2), text, fill=(0, 0, 0), font=font)

        output_filename_img = os.path.basename(os.path.splitext(image_path)[0]) + "_annotated.jpg"
        output_path_img = os.path.join(visualization_dir, output_filename_img)
        
        try:
            image_for_drawing.save(output_path_img)
            print(f"\nVisualization image saved: {output_path_img}")
        except Exception as e:
            print(f"Error: Unable to save visualization image: {e}")
    
    output_filename_json = os.path.basename(os.path.splitext(image_path)[0]) + "_detections.json"
    output_path_json = os.path.join(output_dir, output_filename_json)

    try:
        with open(output_path_json, 'w', encoding='utf-8') as f:
            json.dump(detections, f, indent=4)
        print(f"Detections saved: {output_path_json}")
    except Exception as e:
        print(f"Error: Unable to save results: {e}")

def main():
    if not os.path.isdir(CONFIG["DATA_DIR"]):
        print(f"Error: Dataset directory {CONFIG['DATA_DIR']} does not exist.")
        sys.exit(1)
        
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    os.makedirs(CONFIG["VISUALIZATION_DIR"], exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Using device: {device}")
    print(f"{'='*60}\n")

    print("📦 [Model Loading Stage]")
    try:
        # 1. Spacy
        print(f"\n[1/3] Loading Spacy model...")
        nlp = spacy.load(CONFIG['SPACY_MODEL'])
        print("✓ Spacy model loaded")
        
        # 2. Object Extractor
        print(f"\n[2/3] Loading object extractor...")
        if CONFIG["USE_MISTRAL"]:
            if not os.path.isdir(CONFIG["MISTRAL_MODEL_PATH"]):
                print(f"Mistral model path does not exist, falling back to BERT")
                CONFIG["USE_MISTRAL"] = False
            else:
                extractor = ObjectExtractor(CONFIG["MISTRAL_MODEL_PATH"], device=device, batch_size=4)
        
        if not CONFIG["USE_MISTRAL"]:
            if not os.path.isdir(CONFIG["ZERO_SHOT_BERT_PATH"]):
                print(f"Error: BERT model path does not exist")
                sys.exit(1)
            extractor = pipeline(
                "zero-shot-classification",
                model=CONFIG['ZERO_SHOT_BERT_PATH'],
                device=0 if device == "cuda" else -1
            )
            print("✓ Zero-shot classifier loaded")

        print(f"\n[3/3] Loading detection models...")
        detector = MultiModelDetector(device=device)
        
        detector.load_grounding_dino(CONFIG['GROUNDING_DINO_MODEL'])
        detector.load_owlvit(CONFIG['OWLVIT_MODEL'])
        detector.load_florence2(CONFIG['FLORENCE2_MODEL'])
        
        print("\n✅ All models loaded!\n")
        
    except Exception as e:
        print(f"\nModel loading failed: {e}")
        sys.exit(1)
    
    print(f"{'='*60}")
    print(f"Processing directory: {CONFIG['DATA_DIR']}")
    print(f"{'='*60}")
    
    image_files = [f for f in os.listdir(CONFIG["DATA_DIR"]) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print("Error: No image files found")
        return
    
    print(f"\nFound {len(image_files)} images\n")
    
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print("CUDA acceleration enabled\n")
    
    for idx, image_name in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}]")
        base_name = os.path.splitext(image_name)[0]
        json_name = base_name + ".json"
        
        image_path = os.path.join(CONFIG["DATA_DIR"], image_name)
        json_path = os.path.join(CONFIG["DATA_DIR"], json_name)
        
        if os.path.exists(json_path):
            process_image(
                detector, nlp, extractor,
                image_path, json_path,
                CONFIG["OUTPUT_DIR"],
                CONFIG["VISUALIZATION_DIR"],
                device
            )
        else:
            print(f"Corresponding JSON file not found: {json_name}")
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
