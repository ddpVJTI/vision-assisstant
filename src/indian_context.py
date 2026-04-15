"""
indian_context.py — Real-world sizing map for COCO 80 classes tailored to Indian standard estimates.
All measurements are roughly expected widths in FEET.
"""

INDIAN_OBJECT_WIDTHS_FT = {
    # 🚶‍♂️ People & Accessories
    "person": 1.5,       # Average adult shoulder width
    "backpack": 1.2,
    "umbrella": 3.0,
    "handbag": 1.0,
    "tie": 0.3,
    "suitcase": 1.5,

    # 🚗 Vehicles & Transport
    "bicycle": 2.0,      # Seeing from front/back width
    "car": 5.5,          # Standard Indian hatchback/sedan like Swift
    "motorcycle": 2.5,   # Typical Indian bike width (handles)
    "airplane": 100.0,
    "bus": 8.0,          # City bus
    "train": 10.0,
    "truck": 8.0,
    "boat": 6.0,

    # 🚦 Street & Outdoor Objects
    "traffic light": 1.0,
    "fire hydrant": 1.0,
    "stop sign": 2.0,
    "parking meter": 0.5,
    "bench": 4.0,

    # 🐕 Animals
    "bird": 0.5,
    "cat": 0.8,
    "dog": 1.0,          # Average Indian street dog width
    "horse": 2.0,
    "sheep": 1.5,
    "cow": 2.5,          # Typical Indian cow
    "elephant": 6.0,
    "bear": 3.0,
    "zebra": 2.0,
    "giraffe": 2.0,

    # 🛋️ Furniture & Indoor
    "chair": 1.5,
    "couch": 6.0,
    "potted plant": 1.0,
    "bed": 5.0,
    "dining table": 4.0,
    "toilet": 1.5,

    # 💻 Electronics
    "tv": 3.0,           # Average 32-40 inch TV
    "laptop": 1.2,
    "mouse": 0.2,
    "remote": 0.15,
    "keyboard": 1.5,
    "cell phone": 0.25,

    # 🍳 Kitchen & Appliances
    "microwave": 2.0,
    "oven": 2.0,
    "toaster": 1.0,
    "sink": 2.0,
    "refrigerator": 2.5,

    # 🍔 Food & Drink
    "bottle": 0.3,
    "wine glass": 0.2,
    "cup": 0.3,
    "fork": 0.1,
    "knife": 0.1,
    "spoon": 0.1,
    "bowl": 0.5,
    "banana": 0.5,
    "apple": 0.3,
    "sandwich": 0.4,
    "orange": 0.3,
    "broccoli": 0.3,
    "carrot": 0.4,
    "hot dog": 0.4,
    "pizza": 1.0,
    "donut": 0.3,
    "cake": 0.8,

    # ⚽ Sports & Recreation
    "frisbee": 0.8,
    "skis": 0.4,
    "snowboard": 1.0,
    "sports ball": 0.8,
    "kite": 1.5,
    "baseball bat": 0.3,
    "baseball glove": 1.0,
    "skateboard": 0.8,
    "surfboard": 1.5,
    "tennis racket": 0.8,

    # ✂️ Everyday Items
    "book": 0.6,
    "clock": 1.0,
    "vase": 0.5,
    "scissors": 0.3,
    "teddy bear": 1.0,
    "hair drier": 0.5,
    "toothbrush": 0.1
}

# The default width fallback if something goes horribly wrong
DEFAULT_FALLBACK_WIDTH_FT = 1.0
