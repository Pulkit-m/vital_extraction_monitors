# vital_extraction_monitors
Extracting metrics like heart rate, spo2, respiration rate, etc from images of monitors


Modules: 
* segmentation (Pulkit) 
  * __init__.py 
  * unet
    * __init__.py 
    * train.py 
    * dataset_folder 
    * dataloader.py 
    * infer_masks.py
* object_detection (aman)
  * __init__.py 
  * YOLOv7 
    * __init__.py
    * train.py 
    * dataset_folder
    * infer_bbox.py
* OCR (Aman)
  * __init__.py 
  * detect_ocr.py 
* test_images 
* main.py (Pulkit)
