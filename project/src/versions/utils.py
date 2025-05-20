from pathlib import Path


def is_yolo_model_path(model_path: str) -> bool:
    path = Path(model_path)

    if not path.exists():
        return False
    
    if path.parent.name != "weights":
        return False
    
    if not model_path.endswith(".pt"):
        return False

    # Add more validations here

    return True
