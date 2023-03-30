# vital_extraction_from_monitors
Extracting metrics like heart rate, spo2, respiration rate, etc from images of monitors


# Instructions for running the project

 - Extract the zip file named "*example_name.zip*" to your desired location
 - Create a folder named "*input", inside the  "**inter_iit_main_folder folder*", and load all the test images in this folder 
 - Run all the cells of the "*final_submission.ipynb*" notebook to load all the necessary models and other dependencies.
 - The "*inference" function is present inside the "**final_submission.ipynb*" notebook itself , which accepts an image path string as its only input.
 - Inside "*final_submission.ipynb"  run the cells under the title - "**Testing*" to get all the outputs.
 - The following example illustrates the output of the "*inference*" function:
		{"HR":"88", "SPO2":"98", "RR":"15", "SBP":"126"}
Note: If DBP and MBP are not there in the output dictionary , they are missing
- The "*create_graph_from_image*" digitizes the HR graph and displays it as a `matplotlib` plot. 

If we get "/" in our BP results, then we have split it into SBP and DBP. If not, then we have kept the output as BP.
### We perform the following tasks:
1. Loading the OCR Model
2. Loading the Segmentation and monitor-layout wise classification models.
3. Loading the Yolo object detection models for respective monitor layout types.
4. The *screen_from_segmentation* function takes in the original image as input and returns the warped monitor screen image as output. 
5. The *identify_screen_type* function uses the classificaiton model to classify the monitor screen image into the monitor type.
6. The *create_graph_from_image* function converts the graph image to digitzed output.
7. The *detect_ocr function* identifies characters from the objects detected by the yolo object detection model and labels them into their categories.
8. The *predict* function takes the yolo models and image and provides with the bounding box outputs.
9. The *inference* function implements the entire pipeline.
