# Pyocr

## Traing Ocr model with dataset

```python
from pyocr import OcrModel

ocr_model = OcrModel(
    'dataset-a-z/data/training_data',
    'dataset-a-z/data/testing_data'
)

ocr_model.train_and_save(path="model/model_ocr_v1.model")
ocr_model.save_classes(path="model/classes")
```