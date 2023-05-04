import cv2
import numpy as np
import onnxruntime as ort
#Load the ONNX model
model_file = "models\detection.onnx"
sess = ort.InferenceSession(model_file)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

#Load image
img = cv2.imread("test_image.jpg")

# Preprocess the image to match the model input
input_size = (224, 224) # Change to match your model's input size
preprocessed_image = cv2.resize(img, input_size)
preprocessed_image = preprocessed_image.astype('float32')
preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
preprocessed_image /= 255.0 # normalize pixel values

# Run inference on the preprocessed image using the ONNX model
outputs = sess.run([output_name], {input_name: preprocessed_image})
detections = outputs[0]

# Post-process the output detections and draw them on the image
confidence_threshold = 0.5 # Change to your desired confidence threshold
image_height, image_width, _ = img.shape

# Convert the detections from relative coordinates to absolute coordinates
detections *= np.array([image_width, image_height, image_width, image_height])

# Draw bounding boxes for the detections
for detection in detections:
    x1, y1, x2, y2 = detection.astype(int)
    print(x1,y1,x2, y2)
    print(x1, y1, x2, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the output image
cv2.imshow('Object detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()