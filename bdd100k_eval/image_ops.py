"""Existing native image decoding and YOLO26 tensor/box transformations."""
import base64
import hashlib
import io


NAMES = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motor', 'bike', 'traffic light', 'traffic sign']


def decode_image(value, max_pixels=20_000_000):
    import cv2
    import numpy as np
    from PIL import Image
    if not isinstance(value, str):
        raise ValueError('Trường image phải là Base64')
    data = base64.b64decode(value.split(',', 1)[-1], validate=True)
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > max_pixels:
                raise ValueError('Ảnh có quá nhiều pixel')
            image.verify()
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError('Không thể giải mã dữ liệu hình ảnh') from error
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Không thể giải mã dữ liệu hình ảnh')
    return image, hashlib.sha256(data).hexdigest()


def preprocess(image, fp16=False):
    import cv2
    import numpy as np
    h, w = image.shape[:2]
    ratio = min(640 / w, 640 / h)
    nw, nh = round(w * ratio), round(h * ratio)
    left, top = round((640 - nw) / 2 - .1), round((640 - nh) / 2 - .1)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, 640 - nh - top, left, 640 - nw - left,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None])
    return tensor.astype(np.float16 if fp16 else np.float32) / 255, (ratio, left, top, w, h)


def postprocess(output, transform, conf):
    import numpy as np
    output = np.asarray(output)
    if output.shape != (1, 300, 6):
        raise RuntimeError(f'Expected YOLO26 end-to-end [1,300,6], got {output.shape}')
    ratio, left, top, w, h = transform
    boxes = []
    for row in output[0]:
        if not np.isfinite(row).all() or row[4] < conf or row[4] <= 0:
            continue
        cls = int(row[5])
        if not 0 <= cls < len(NAMES) or row[5] != cls:
            raise RuntimeError('Invalid BDD100K class id')
        coords = [(float(row[i]) - (left if i % 2 == 0 else top)) / ratio for i in range(4)]
        coords = [max(0., min(w if i % 2 == 0 else h, x)) for i, x in enumerate(coords)]
        boxes.append(dict(class_id=cls, class_name=NAMES[cls], conf=float(row[4]), box_px=coords))
    return boxes
