# Shinigami no me
Shinigami no me is a simple package to do optical character recognition with python.
This package is made with python with [tensorflow](https://www.tensorflow.org) develop by google ,
and [opencv-python](https://docs.opencv.org).
this package uses CNN (convolution neuron network) to do image recognition

## Build this package

```bash
python -m pip install --upgrade build
python -m build
```

## Install package

```bash
python -m pip install dist/shinigami_no_me-0.0.1-py3-none-any.whl
```

## Train your own Ocr model with dataset
* #### Example
```python
from shinigami_no_me import OcrModel

ocr_model = OcrModel(
    'dataset-a-z/data/training_data',
    'dataset-a-z/data/testing_data'
)

ocr_model.train_and_save(path="model_ocr_v1.model")
ocr_model.save_classes(path="classes")
```
