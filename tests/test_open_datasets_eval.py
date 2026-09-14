import json
from pathlib import Path

from PIL import Image

from src.datasets.coco_caption_dataset import CocoCaptionDataset
from src.eval.chair_eval import convert_to_chair_format, run_chair_eval
from src.eval.mmhal_format import convert_to_mmhal_response


def test_coco_caption_dataset_loads_val2014_schema(tmp_path):
    image_root, ann_root = _make_coco_fixture(tmp_path)

    dataset = CocoCaptionDataset(
        image_root=str(image_root),
        annotation_root=str(ann_root),
        limit=1,
        prompt="caption prompt",
    )

    assert len(dataset) == 1
    item = dataset[0]
    assert item["question_id"] == "1"
    assert item["image_id"] == 1
    assert item["image_name"] == "COCO_val2014_000000000001.jpg"
    assert item["question"] == "caption prompt"
    assert item["task"] == "chair"


def test_chair_eval_wrapper_outputs_core_metrics(tmp_path):
    image_root, ann_root = _make_coco_fixture(tmp_path)
    _ = image_root
    result_path = tmp_path / "results_chair_greedy.json"
    result_path.write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "caption": "A dog beside a cat.",
                    "best_answer": "A dog beside a cat.",
                }
            ]
        ),
        encoding="utf-8",
    )
    chair_path = tmp_path / "chair_format_greedy.json"
    metrics_path = tmp_path / "metrics_chair_greedy.json"

    convert_to_chair_format(str(result_path), str(chair_path))
    metrics = run_chair_eval(str(chair_path), str(ann_root), str(metrics_path))

    assert metrics["N"] == 1
    assert "CHAIRs" in metrics
    assert "CHAIRi" in metrics
    assert "Recall" in metrics
    assert "AvgLen" in metrics
    assert metrics_path.exists()


def test_mmhal_response_format_preserves_template_fields(tmp_path):
    template = [
        {
            "image_id": "abc",
            "question": "What is visible?",
            "gt_answer": "A dog.",
            "image_content": ["dog"],
        }
    ]
    template_path = tmp_path / "response_template.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")
    result_path = tmp_path / "results_mmhal_greedy.json"
    result_path.write_text(
        json.dumps([{"question_id": "0", "best_answer": "A dog is visible."}]),
        encoding="utf-8",
    )
    output_path = tmp_path / "mmhal_response_greedy.json"

    rows = convert_to_mmhal_response(str(result_path), str(output_path), str(template_path))

    assert rows[0]["image_id"] == "abc"
    assert rows[0]["question_id"] == "0"
    assert rows[0]["model_answer"] == "A dog is visible."


def _make_coco_fixture(tmp_path: Path):
    image_root = tmp_path / "val2014"
    ann_root = tmp_path / "annotations"
    image_root.mkdir()
    ann_root.mkdir()
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(
        image_root / "COCO_val2014_000000000001.jpg"
    )
    instances = {
        "images": [{"id": 1, "file_name": "COCO_val2014_000000000001.jpg"}],
        "categories": [{"id": 18, "name": "dog"}],
        "annotations": [{"image_id": 1, "category_id": 18}],
    }
    captions = {
        "images": [{"id": 1, "file_name": "COCO_val2014_000000000001.jpg"}],
        "annotations": [{"image_id": 1, "caption": "A dog on grass."}],
    }
    (ann_root / "instances_val2014.json").write_text(json.dumps(instances), encoding="utf-8")
    (ann_root / "captions_val2014.json").write_text(json.dumps(captions), encoding="utf-8")
    return image_root, ann_root
