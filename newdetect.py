from ultralytics import YOLO
import cv2

model = YOLO('C:/object-detection-realtime/best-new.pt')

image_folder = 'C:/object-detection-realtime/image-detection/images'
output_folder = 'C:/object-detection-realtime/image-detection/results'

import os
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
   
    results = model.predict(source=image_path, save=True, save_dir=output_folder)

    result_image = results[0].plot()
    cv2.imshow('Detection Results', result_image)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):  
        break

cv2.destroyAllWindows()
