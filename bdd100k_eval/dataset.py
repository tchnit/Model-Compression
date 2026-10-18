"""BDD100K class mapping, annotations and existing YOLO label conversion."""
import json
import math
from pathlib import Path
import random
import shutil
from .artifacts import sha256
from .reporting import write_json


DEFAULT_NAMES = ["pedestrian", "rider", "car", "truck", "bus", "train",
                 "motorcycle", "bicycle", "traffic light", "traffic sign"]


ALIASES = {"person": "pedestrian", "bike": "bicycle", "motor": "motorcycle"}


def canonical(name):
    name = str(name).strip().lower().replace("_", " ")
    return ALIASES.get(name, name)


def read_frames(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("frames"), list):
        data = data["frames"]
    if not isinstance(data, list):
        raise ValueError("Nhãn phải là list frame BDD100K/Scalabel hoặc object có frames; không phải COCO JSON.")
    return data


def prepare_dataset(args, out):
    from PIL import Image
    import yaml

    images_dir = Path(args.images).expanduser().resolve()
    if not images_dir.is_dir():
        raise ValueError(f"Không tìm thấy thư mục ảnh: {images_dir}")
    frames = read_frames(args.labels)
    by_name = {}
    for frame in frames:
        name = frame.get("name")
        if not name or Path(name).name != name:
            raise ValueError("Frame name phải là tên ảnh, không chứa đường dẫn.")
        if name in by_name:
            raise ValueError(f"Nhãn trùng ảnh: {name}")
        by_name[name] = frame
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise ValueError("Không có ảnh.")
    missing = [p.name for p in images if p.name not in by_name]
    if missing:
        raise ValueError(f"{len(missing)} ảnh thiếu entry annotation; ví dụ: {missing[:5]}")
    if len({p.stem for p in images}) != len(images):
        raise ValueError("Tên stem ảnh trùng, sẽ ghi đè nhãn YOLO.")
    if args.limit:
        images = sorted(random.Random(args.seed).sample(images, min(args.limit, len(images))))
    names = [canonical(n) for n in args.names.split(",")]
    if len(names) != 10 or set(names) != set(DEFAULT_NAMES):
        raise ValueError("--names phải chứa đúng 10 lớp BDD100K, theo thứ tự lớp checkpoint.")
    mapping = {name: i for i, name in enumerate(names)}
    dataset = out / "dataset"
    image_out, label_out = dataset / "images/val", dataset / "labels/val"
    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)
    stats = {"images": len(images), "empty_images": 0, "boxes": 0, "unknown_box_categories": {},
             "ignore_or_crowd_boxes": 0, "clipped_boxes": 0, "degenerate_boxes": 0,
             "objects_per_class": {n: 0 for n in names}}
    manifest = []
    max_objects = 0
    for image in images:
        with Image.open(image) as im:
            width, height = im.size
        frame = by_name[image.name]
        if "labels" not in frame or frame["labels"] is None:
            raise ValueError(f"{image.name}: thiếu labels; ảnh không có vật thể cần labels: [].")
        rows = []
        for obj in frame["labels"]:
            box = obj.get("box2d")
            if box is None:
                continue  # lanes, drivable-area, and other non-detection labels
            category = canonical(obj.get("category", ""))
            if category not in mapping:
                unknown = stats["unknown_box_categories"]
                unknown[category] = unknown.get(category, 0) + 1
                continue
            attrs = obj.get("attributes") or {}
            ignored = any(attrs.get(k) is True or attrs.get(k) == 1
                          for k in ("ignore", "ignored", "crowd", "iscrowd"))
            if ignored:
                stats["ignore_or_crowd_boxes"] += 1
                if args.ignore_policy == "error":
                    raise ValueError("Có ignore/crowd: YOLO txt không biểu diễn được. Xem --ignore-policy.")
                if args.ignore_policy == "drop":
                    continue
            raw = [float(box[k]) for k in ("x1", "y1", "x2", "y2")]
            if not all(math.isfinite(v) for v in raw):
                raise ValueError(f"Box không hữu hạn: {image.name}")
            x1, y1, x2, y2 = raw
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Box đảo hoặc rỗng: {image.name}, {raw}")
            clipped = [max(0, min(width, x1)), max(0, min(height, y1)),
                       max(0, min(width, x2)), max(0, min(height, y2))]
            stats["clipped_boxes"] += int(clipped != raw)
            x1, y1, x2, y2 = clipped
            if x2 <= x1 or y2 <= y1:
                stats["degenerate_boxes"] += 1
                continue
            rows.append(f"{mapping[category]} {(x1+x2)/(2*width):.9f} {(y1+y2)/(2*height):.9f} "
                        f"{(x2-x1)/width:.9f} {(y2-y1)/height:.9f}")
            stats["objects_per_class"][category] += 1
        label = label_out / (image.stem + ".txt")
        label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        target = image_out / image.name
        try:
            target.symlink_to(image)
        except OSError:
            shutil.copy2(image, target)
        stats["empty_images"] += int(not rows)
        stats["boxes"] += len(rows)
        max_objects = max(max_objects, len(rows))
        manifest.append({"name": image.name, "width": width, "height": height,
                         "image_bytes": image.stat().st_size, "image_sha256": sha256(image),
                         "label_sha256": sha256(label)})
    if stats["unknown_box_categories"]:
        raise ValueError(f"Có box category ngoài 10 lớp; cần kiểm tra nguồn nhãn: {stats['unknown_box_categories']}")
    if not stats["boxes"]:
        raise ValueError("Không có ground-truth box.")
    if max_objects > args.max_det:
        raise ValueError(f"Có ảnh chứa {max_objects} objects > max_det={args.max_det}; tăng --max-det và kiểm tra graph export.")
    yaml_path = dataset / "bdd100k.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": str(dataset), "train": "images/val", "val": "images/val",
                                          "names": names}, allow_unicode=True), encoding="utf-8")
    write_json(out / "dataset_manifest.json", manifest)
    write_json(out / "dataset_stats.json", stats)
    return yaml_path, names, [str(image_out / p.name) for p in images], stats
