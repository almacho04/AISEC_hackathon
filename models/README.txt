Project: Stamp & Signature Detection  
Model: best_yolo11_stamp_signature_IDP.pt  
Date: [YYYY-MM-DD]

1. Purpose  
   This model detects two object classes in scanned document images:  
   • signature  
   • stamp  

2. Input  
   - A single image (JPEG, PNG) or a page extracted from a PDF (pre-converted to image)  
   - Recommended image size: up to ~1024 px width (trained with imgsz=1024)  
   - Preprocessing:  
     • Pages were converted from PDF to images (via PyMuPDF)  
     • Images down-scaled if width > 2000 px to speed inference  
     • Full pages used — no manual cropping required  

3. Output  
   - For each input image, the model outputs bounding boxes for each detected object of class “signature” or “stamp”  
   - Output format:  
     • Bounding box coordinates: (x1, y1, x2, y2) + class id + confidence score  
     • Visual overlay: boxes drawn on the original image, color-coded by class  
   - Example usage:  
     ```python
     from ultralytics import YOLO
     model = YOLO("models/best_yolo11_stamp_signature_IDP.pt")
     results = model.predict("page1.jpg", imgsz=1024, conf=0.25)
     results.show()  # visualises boxes
     ```

4. Training details  
   - Dataset: IDP_stamp_signature_detection.v2i.yolov8 (labels in YOLO format)  
   - Classes:  
       0 → signature  
       1 → stamp  
   - Training configuration:  
       • Image resolution (imgsz): 1024  
       • Batch size: 8 (2 GPUs)  
       • Epochs: 5 (initial run)  
       • Multi-GPU devices: [2, 3]  
   - Run directory: `runs/yolo_stamp_signature_IDP`  
   - Best weights stored at: `models/best_yolo11_stamp_signature_IDP.pt`  

5. Inference & deployment  
   - Use the weights file above for inference or export to other formats (ONNX, TensorRT) using:  
     ```python
     model.export(format="onnx")
     ```  
   - For real-time or batch document scanning pipelines: load the model, run `predict()` on each page image, then parse bounding boxes for further processing (e.g., crop the stamp area, extract the signature, etc.).

6. Notes & caveats  
   - This model was fine-tuned on a custom dataset of document images; performance may vary on very different domains (e.g., different languages, low-quality scans).  
   - Overlapping stamps and signatures may still require additional training or post-processing logic to separate them reliably.  
   - On large document batches or high-resolution images, ensure GPU memory is sufficient – otherwise reduce `imgsz` or batch size.

7. Contact  
   - If you use this model in your hackathon/project, please cite: [Your Name], AISEC Hackathon 2025 (Kazakhstan)  
