from ultralytics import YOLO
import warnings

# Initialize the YOLO model
model = YOLO("yolov8n.pt")

# Define search space
# search_space = {
#     "lr0": (1e-5, 1e-1),
#     "lrf": (0.01, 1.0),
#     "momentum": (0.6, 0.9),
#     "weight_decay": (0.0, 1e-3),
#     "warmup_epochs": (0.0, 5.0),
#     "warmup_momentum": (0.0, 0.95),
#     "box": (0.02, 0.2),
#     "cls": (0.2, 4.0),
#     "hsv_h": (0.0, 0.3),
#     "hsv_s": (0.0, 0.9),
#     "hsv_v": (0.0, 0.9),
#     "degrees": (1e-5, 20.0),
#     "translate": (0.0, 0.9),
#     "scale": (0.0, 0.3),
#     "shear": (1e-5, 10.0),
#     "perspective": (1e-5, 0.001),
#     "flipud": (0.0, 0.0),
#     "fliplr": (0.0, 0.0),
#     "mosaic": (0.0, 0.0),
#     "mixup": (1e-5, 1.0),
#     "copy_paste": (0.0, 0.0),
# }

# Search space around optimal hyperparameters from tuning 1
search_space = {
    "lr0": (0.0035, 0.007),  # Learning rate, ±30%
    "lrf": (0.008, 0.015),  # Final learning rate fraction, ±35%
    "momentum": (0.85, 0.93),  # Momentum, ±5%
    "weight_decay": (0.00015, 0.0004),  # Weight decay, ±40%
    "warmup_epochs": (2.0, 4.0),  # Warmup epochs, ±30%
    "warmup_momentum": (0.8, 0.9),  # Warmup momentum, ±6%
    "box": (0.15, 0.25),  # Box loss gain, ±25%
    "cls": (0.18, 0.28),  # Classification loss gain, ±25%
    "hsv_h": (0.008, 0.025),  # Hue augmentation, ±40%
    "hsv_s": (0.5, 0.85),  # Saturation augmentation, ±25%
    "hsv_v": (0.2, 0.4),  # Value augmentation, ±30%
    "degrees": (0.0, 0.005),  # Small rotation range
    "translate": (0.08, 0.18),  # Translation, ±30%
    "scale": (0.2, 0.4),  # Scale augmentation, ±33%
    "shear": (0.0, 0.005),  # Small shear range
    "perspective": (0.0, 0.005),  # Small perspective range
    "flipud": (0.0, 0.1),  # Allowing some vertical flips
    "fliplr": (0.0, 0.1),  # Allowing some horizontal flips
    "mosaic": (0.0, 0.2),  # Keeping mosaic small
    "mixup": (0.0, 0.005),  # Small mixup range
    "copy_paste": (0.0, 0.1),  # Allowing copy-paste
}

# Tune hyperparameters on COCO8 for 50 epochs
model.tune(
    data="data.yaml",
    epochs=50,
    iterations=300,
    optimizer="AdamW",
    space=search_space,
    plots=False,
    save=False,
    val=False,
    device="cuda",
)