from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(
    data="/mnt/sda1/home/bmestaging/long_cam/Blood_Cell/datasets/Alam2019/data.yaml", 
    epochs=20, 
    project="/mnt/sda1/home/bmestaging/long_cam/Blood_Cell/notebook/Alam2019/runs/detect", name="train"
    )

