# Models

Para exportar el modelo se usó:

``` console
yolo export model=yolov8s-seg.pt imgsz=640 format=onnx opset=12 simplify device=0
```

Toda la documentación respecto a `yolo export` se encuentra [aquí](https://docs.ultralytics.com/modes/export/), para .los cuales se obtienen los modelos que están en [gpu](./gpu/). Cambiando el parametro `device=cpu`, están los modelos descargados en [cpu](./cpu/).
