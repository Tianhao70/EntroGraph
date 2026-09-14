from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.eval.caption_metrics import word_count


try:
    from playground.chair.chair import synonyms_txt
except Exception:  # pragma: no cover
    synonyms_txt = ""


COCO_DOUBLE_WORDS = [
    "motor bike",
    "motor cycle",
    "air plane",
    "traffic light",
    "street light",
    "traffic signal",
    "stop light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "suit case",
    "sports ball",
    "baseball bat",
    "baseball glove",
    "tennis racket",
    "wine glass",
    "hot dog",
    "cell phone",
    "mobile phone",
    "teddy bear",
    "hair drier",
    "potted plant",
    "bow tie",
    "laptop computer",
    "stove top oven",
    "home plate",
    "train track",
]
ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal", "cub"]
VEHICLE_WORDS = ["jet", "train"]
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def convert_to_chair_format(input_json: str, output_json: str) -> None:
    with open(input_json, "r", encoding="utf-8") as f:
        rows = json.load(f)
    converted = []
    for item in rows:
        caption = item.get("caption", item.get("best_answer", ""))
        converted.append({"image_id": int(item["image_id"]), "caption": str(caption)})
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)


def run_chair_eval(
    cap_file: str,
    annotation_root: str,
    output_json: str,
) -> dict[str, Any]:
    cache_path = os.path.join(annotation_root, "entrograph_chair_val2014.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            evaluator = pickle.load(f)
    else:
        evaluator = ChairEvaluator(annotation_root)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(evaluator, f)
        except OSError:
            pass

    details = evaluator.compute_chair(cap_file)
    overall = details["overall_metrics"]
    metrics = {
        "method": _method_from_path(cap_file),
        "N": details["N"],
        "CHAIRs": overall["CHAIRs"],
        "CHAIRi": overall["CHAIRi"],
        "Recall": overall["Recall"],
        "Precision": overall["Precision"],
        "F1": overall["F1"],
        "AvgLen": overall["AvgLen"],
        "ObjMentioned": overall["ObjMentioned"],
        "chair_detail_path": output_json,
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, **details}, f, ensure_ascii=False, indent=2)
    return metrics


class ChairEvaluator:
    def __init__(self, annotation_root: str):
        self.annotation_root = Path(annotation_root).expanduser()
        self.instances_path = self.annotation_root / "instances_val2014.json"
        self.captions_path = self.annotation_root / "captions_val2014.json"
        if not self.instances_path.exists():
            raise FileNotFoundError(f"Missing instances_val2014.json in {self.annotation_root}")
        if not self.captions_path.exists():
            raise FileNotFoundError(f"Missing captions_val2014.json in {self.annotation_root}")

        self.mscoco_objects: list[str] = []
        self.inverse_synonym_dict: dict[str, str] = {}
        self.double_word_dict: dict[str, str] = {}
        self.imid_to_objects: dict[int, set[str]] = defaultdict(set)
        self._init_synonyms()
        self._load_annotations()

    def _init_synonyms(self) -> None:
        synonyms = [line.strip().split(", ") for line in synonyms_txt.splitlines() if line.strip()]
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for word in synonym:
                self.inverse_synonym_dict[word] = synonym[0]

        for double_word in COCO_DOUBLE_WORDS:
            self.double_word_dict[double_word] = double_word
        for animal_word in ANIMAL_WORDS:
            self.double_word_dict[f"baby {animal_word}"] = animal_word
            self.double_word_dict[f"adult {animal_word}"] = animal_word
        for vehicle_word in VEHICLE_WORDS:
            self.double_word_dict[f"passenger {vehicle_word}"] = vehicle_word
        self.double_word_dict["bow tie"] = "tie"
        self.double_word_dict["toilet seat"] = "toilet"
        self.double_word_dict["wine glas"] = "wine glass"

    def _load_annotations(self) -> None:
        with self.instances_path.open("r", encoding="utf-8") as f:
            instances = json.load(f)
        cat_id_to_name = {cat["id"]: cat["name"] for cat in instances.get("categories", [])}
        for annotation in instances.get("annotations", []):
            category = cat_id_to_name.get(annotation.get("category_id"))
            if category in self.inverse_synonym_dict:
                self.imid_to_objects[int(annotation["image_id"])].add(self.inverse_synonym_dict[category])

        with self.captions_path.open("r", encoding="utf-8") as f:
            captions = json.load(f)
        for annotation in captions.get("annotations", []):
            _, node_words, _, _ = self.caption_to_words(annotation.get("caption", ""))
            self.imid_to_objects[int(annotation["image_id"])].update(node_words)

    def caption_to_words(self, caption: str) -> tuple[list[str], list[str], list[int], list[str]]:
        words = self._tokenize_and_lemmatize(caption)
        collapsed_words: list[str] = []
        idxs: list[int] = []
        i = 0
        while i < len(words):
            idxs.append(i)
            double_word = " ".join(words[i : i + 2])
            if double_word in self.double_word_dict:
                collapsed_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                collapsed_words.append(words[i])
                i += 1

        if "toilet" in collapsed_words and "seat" in collapsed_words:
            collapsed_words = [word for word in collapsed_words if word != "seat"]

        idxs = [idx for idx, word in enumerate(collapsed_words) if word in self.inverse_synonym_dict]
        words = [word for word in collapsed_words if word in self.inverse_synonym_dict]
        node_words = [self.inverse_synonym_dict[word] for word in words]
        return words, node_words, idxs, collapsed_words

    def _tokenize_and_lemmatize(self, caption: str) -> list[str]:
        try:
            import nltk
            from nltk.corpus import wordnet
            from nltk.stem import WordNetLemmatizer

            raw_tokens = nltk.word_tokenize(caption.lower())
            tagged = nltk.pos_tag(raw_tokens)
            lemmatizer = WordNetLemmatizer()
            lemmas = []
            for token, tag in tagged:
                pos = wordnet.NOUN
                if tag.startswith("J"):
                    pos = wordnet.ADJ
                elif tag.startswith("V"):
                    pos = wordnet.VERB
                elif tag.startswith("R"):
                    pos = wordnet.ADV
                lemmas.append(lemmatizer.lemmatize(token, pos=pos))
            return lemmas
        except Exception:
            tokens = WORD_RE.findall(caption.lower())
            return [_simple_singular(token) for token in tokens]

    def compute_chair(self, cap_file: str) -> dict[str, Any]:
        with open(cap_file, "r", encoding="utf-8") as f:
            captions = json.load(f)
        if not isinstance(captions, list):
            raise ValueError(f"CHAIR cap file must be a list: {cap_file}")

        num_caps = 0
        num_hallucinated_caps = 0
        hallucinated_word_count = 0
        coco_word_count = 0
        len_caps = 0
        num_recall_gt_objects = 0
        num_gt_objects = 0
        num_generated_objects = 0
        sentences = []

        for row in captions:
            image_id = int(row["image_id"])
            caption = str(row.get("caption", ""))
            words, node_words, idxs, raw_words = self.caption_to_words(caption)
            gt_objects = self.imid_to_objects.get(image_id, set())
            generated_set = set(node_words)
            recall_gt_objects = set()
            hallucinated_words = []
            hallucination_idxs = []

            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word not in gt_objects:
                    hallucinated_word_count += 1
                    hallucinated_words.append((word, node_word))
                    hallucination_idxs.append(idx)
                else:
                    recall_gt_objects.add(node_word)

            hallucinated = bool(hallucinated_words)
            num_caps += 1
            num_hallucinated_caps += int(hallucinated)
            coco_word_count += len(node_words)
            len_caps += word_count(caption)
            num_gt_objects += len(gt_objects)
            num_generated_objects += len(generated_set)
            num_recall_gt_objects += len(recall_gt_objects)

            precision = _safe_div(len(recall_gt_objects), len(generated_set))
            recall = _safe_div(len(recall_gt_objects), len(gt_objects))
            sentences.append(
                {
                    "image_id": image_id,
                    "caption": caption,
                    "mscoco_hallucinated_words": hallucinated_words,
                    "mscoco_gt_words": sorted(gt_objects),
                    "mscoco_words": words,
                    "mscoco_generated_words": sorted(generated_set),
                    "hallucination_idxs": hallucination_idxs,
                    "words": raw_words,
                    "metrics": {
                        "CHAIRs": int(hallucinated),
                        "CHAIRi": _safe_div(len(hallucinated_words), len(words)),
                        "Recall": recall,
                        "Precision": precision,
                        "F1": _safe_f1(precision, recall),
                        "Len": word_count(caption),
                    },
                }
            )

        recall = _safe_div(num_recall_gt_objects, num_gt_objects)
        precision = _safe_div(num_recall_gt_objects, num_generated_objects)
        return {
            "N": num_caps,
            "overall_metrics": {
                "CHAIRs": _safe_div(num_hallucinated_caps, num_caps),
                "CHAIRi": _safe_div(hallucinated_word_count, coco_word_count),
                "Recall": recall,
                "Precision": precision,
                "F1": _safe_f1(precision, recall),
                "AvgLen": _safe_div(len_caps, num_caps),
                "ObjMentioned": _safe_div(num_generated_objects, num_caps),
            },
            "sentences": sentences,
        }


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _safe_f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _simple_singular(token: str) -> str:
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _method_from_path(path: str) -> str:
    stem = Path(path).stem
    for prefix in ("chair_format_", "results_chair_", "metrics_chair_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CHAIR evaluation for EntroGraph caption outputs.")
    parser.add_argument("--input", required=True, help="EntroGraph result JSON or CHAIR format JSON.")
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chair-format-output", default=None)
    args = parser.parse_args()

    input_path = args.input
    if not Path(input_path).name.startswith("chair_format_"):
        chair_format_output = args.chair_format_output
        if chair_format_output is None:
            method = _method_from_path(input_path)
            chair_format_output = str(Path(input_path).with_name(f"chair_format_{method}.json"))
        convert_to_chair_format(input_path, chair_format_output)
        input_path = chair_format_output

    metrics = run_chair_eval(input_path, args.annotation_root, args.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

