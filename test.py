import cv2
import torch
from torchvision import transforms

# Load the YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='D:/syed/yolov5/runs/train/exp4/weights/best.pt').to(device).eval()

# Set up the webcam capture
cap = cv2.VideoCapture(0)  # 0 represents the default webcam device

# Enter a loop to continuously capture frames and perform inference
while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Preprocess the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame = transforms.ToTensor()(frame).unsqueeze(0).to(device)  # Convert to tensor and move to device

    # Perform inference
    results = model(frame)

    # Process the results (e.g., draw bounding boxes)
    # ...

    # Display the frame with results
    cv2.imshow('Webcam', frame)

    # Check for key press to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()
