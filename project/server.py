import cv2
import numpy as np

from main import main
from src.config import settings
from src.versions import version_manager, classes

from typing import Annotated
from fastapi import FastAPI, File
from fastapi.responses import Response
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def top5_cls_func(cls_id: list[int], conf: list[float]) -> int:
    """
    Function to apply top-5 results in order to get the class id.
    """
    return int(sum((i * conf_i) for i, conf_i in zip(cls_id, conf)))


def top5_conf_func(cls_id: list[int], conf: list[float]) -> float:
    """
    Function to apply top-5 results in order to get the resultant confidence.
    """
    return max(conf)


@app.get("/")
async def root() -> dict:
    return {"message": "Hello World"}


@app.post("/braille")
def translate_braille(
    image: Annotated[bytes, File()],
    character_model_version: str,
    translation_model_version: str,
    top1: bool = True,
):
    """
    Process an image to detect and translate Braille characters.
    """
    if isinstance(image, bytes):
        image_cv2 = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR_BGR)

    if character_model_version not in version_manager.list_versions("characters"):
        return HTTPException(
            status_code=404, detail="Character model version not found"
        )

    if translation_model_version not in version_manager.list_versions("translation"):
        return HTTPException(
            status_code=404, detail="Translation model version not found"
        )

    if character_model_version == "v1":
        characters_kwargs: classes.YOLOInput = {
            "yolo_model_path": "./models/runs/detect/train2/weights/best.pt",
            "version": character_model_version,
            "conf": 0.7,
            "iou": 0.7,
        }


    if translation_model_version == "v1":
        translation_kwargs: classes.PytorchTranslationInput = {
            "translation_model_path": "./models/runs/translation/train4/best_model_epoch92.pth",
            "version": translation_model_version,
            "device": "cpu",
        }

    if translation_model_version == "v2":
        translation_kwargs: classes.YOLOInput = {
            "yolo_model_path": "./models/runs/translation/train5-yolo/weights/best.pt",
            "version": translation_model_version,
            "conf": 0.7,
            "iou": 0.7,
        }

    extra_kwargs = {
        "top1": top1,
        "top5_cls_func": top5_cls_func,
        "top5_conf_func": top5_conf_func,
    }

    result: classes.OutputPrediction | None = main(
        image=image_cv2,
        characters_kwargs=characters_kwargs,
        translation_kwargs=translation_kwargs,
        **extra_kwargs,
    )

    if result is None:
        raise HTTPException(status_code=500, detail="Error processing the image")

    result_img = result["result_img"]
    _, img_encoded = cv2.imencode('.png', result_img)

    return Response(content=img_encoded.tobytes(), media_type="image/png")
