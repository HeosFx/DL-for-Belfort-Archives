from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolo11n-cls.pt")

# Define search space

# Tuning 1
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
#     "degrees": (0.0, 20.0, 10.0),
#     "translate": (0.0, 0.9, 0.4),
#     "scale": (0.0, 0.3, 0.1),
#     "shear": (0.0, 10.0, 5.0),
#     "perspective": (0.0, 0.001, 0.001),
#     "flipud": (0.0, 0.5, 0.25),
#     "fliplr": (0.0, 0.5, 0.25),
#     "mosaic": (0.0, 0.5, 0.25),
#     "mixup": (0.0, 1.0, 0.5),
#     "copy_paste": (0.0, 0.0),
# }


# Tuning 2
# search_space = {
#     "lr0": (0.008, 0.013),  # Learning rate, ±25%
#     "lrf": (0.0085, 0.0135),  # Final learning rate fraction, ±25%
#     "momentum": (0.83, 0.90),  # Momentum, ±5%
#     "weight_decay": (0.00035, 0.00065),  # Weight decay, ±30%
#     "warmup_epochs": (2.7, 4.0),  # Warmup epochs, ±20%
#     "warmup_momentum": (0.90, 0.95),  # Warmup momentum, ±5%
#     "box": (0.15, 0.23),  # Box loss gain, ±20%
#     "cls": (0.47, 0.71),  # Classification loss gain, ±20%
#     "hsv_h": (0.01, 0.02),  # Hue augmentation, ±30%
#     "hsv_s": (0.55, 0.80),  # Saturation augmentation, ±20%
#     "hsv_v": (0.35, 0.52),  # Value augmentation, ±20%
#     "degrees": (0.0, 0.005),  # Small rotation range
#     "translate": (0.07, 0.12),  # Translation, ±25%
#     "scale": (0.23, 0.37),  # Scale augmentation, ±25%
#     "shear": (0.0, 0.005),  # Small shear range
#     "perspective": (0.0, 0.005),  # Small perspective range
#     "flipud": (0.0, 0.05),  # Allowing small vertical flips
#     "fliplr": (0.40, 0.60),  # Horizontal flip probability, ±20%
#     "mosaic": (0.40, 0.60),  # Mosaic augmentation, ±20%
#     "mixup": (0.0, 0.05),  # Allowing small mixup probability
#     "copy_paste": (0.0, 0.05),  # Allowing small copy-paste probability
# }

# # Tuning 3
search_space = {
    "lr0": (1e-5, 1e-1),  # Learning rate
    "lrf": (0.01, 1.0),  # Final learning rate fraction
    "momentum": (0.6, 0.98),  # Momentum (slightly higher upper bound)
    "weight_decay": (1e-5, 1e-3),  # Avoids 0 for better exploration
    "warmup_epochs": (0.1, 5.0),  # Avoids 0 to ensure some warmup
    "warmup_momentum": (0.1, 0.95),  # Avoids 0 for smoother transition
    "box": (0.05, 0.3),  # Slightly higher range
    "cls": (0.2, 4.0),  # Classification loss gain
    "hsv_h": (0.00001, 0.3),  # Avoids 0 for some hue augmentation
    "hsv_s": (0.00001, 0.9),  # Avoids 0 for saturation adjustment
    "hsv_v": (0.00001, 0.9),  # Avoids 0 for value adjustment
    "degrees": (1.0, 20.0),  # Avoids 0-degree rotations
    "translate": (0.1, 0.9),  # Avoids 0 for some translation
    "scale": (0.05, 0.3),  # Ensures minimal scaling occurs
    "shear": (1.0, 10.0),  # Avoids no shear
    "perspective": (1e-5, 0.001),  # Avoids 0
    "flipud": (0.05, 0.5),  # Avoids 0 for some vertical flips
    "fliplr": (0.05, 0.5),  # Avoids 0 for some horizontal flips
    "mosaic": (0.1, 0.5),  # Ensures some mosaic augmentation
    "mixup": (0.1, 1.0),  # Ensures mixup is tested
    "copy_paste": (0.1, 0.5),  # If supported, ensures some level of use
}

# Tuning 4 - no hsv
# search_space = {
#     "lr0": (1e-5, 1e-1),  # Learning rate
#     "lrf": (0.01, 1.0),  # Final learning rate fraction
#     "momentum": (0.6, 0.98),  # Momentum (slightly higher upper bound)
#     "weight_decay": (1e-5, 1e-3),  # Avoids 0 for better exploration
#     "warmup_epochs": (0.1, 5.0),  # Avoids 0 to ensure some warmup
#     "warmup_momentum": (0.1, 0.95),  # Avoids 0 for smoother transition
#     "box": (0.05, 0.3),  # Slightly higher range
#     "cls": (0.2, 4.0),  # Classification loss gain
#     "hsv_h": (0.00001, 0.00002),  # Avoids 0 for some hue augmentation
#     "hsv_s": (0.00001, 0.00002),  # Avoids 0 for saturation adjustment
#     "hsv_v": (0.00001, 0.00002),  # Avoids 0 for value adjustment
#     "degrees": (1.0, 20.0),  # Avoids 0-degree rotations
#     "translate": (0.1, 0.9),  # Avoids 0 for some translation
#     "scale": (0.05, 0.3),  # Ensures minimal scaling occurs
#     "shear": (1.0, 10.0),  # Avoids no shear
#     "perspective": (1e-5, 0.001),  # Avoids 0
#     "flipud": (0.05, 0.5),  # Avoids 0 for some vertical flips
#     "fliplr": (0.05, 0.5),  # Avoids 0 for some horizontal flips
#     "mosaic": (0.1, 0.5),  # Ensures some mosaic augmentation
#     "mixup": (0, 1.0),  # Ensures mixup is tested
#     "copy_paste": (0.1, 0.5),  # If supported, ensures some level of use
# }

# Tune hyperparameters on dataset for 50 epochs
model.tune(
    data="./dataset_color",
    epochs=50,
    iterations=300,
    batch=16,
    optimizer="AdamW",
    space=search_space,
    plots=False,
    save=False,
    val=False,
    device="cuda",
    imgsz=1024,
    workers=16,
    name="train_tuning_yolo11n_cls",
)