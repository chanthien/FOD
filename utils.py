import cv2
import numpy as np

def load_model(model_path):
    from ultralytics import YOLO
    model = YOLO(model_path)
    return model

def detect_objects(model, frame):
    results = model([frame])
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy()  # confidence scores
        for box, conf in zip(boxes, confidences):
            detections.append({
                'id': None,
                'name': 'object',
                'bbox': box,
                'confidence': conf
            })
    return detections

def compute_perspective_transform(gps_points, image_points):
    matrix, _ = cv2.findHomography(np.array(image_points), np.array(gps_points))
    return matrix

def transform_to_gps(detections, matrix):
    transformed_detections = []
    for det in detections:
        points = np.array([[
            [det['bbox'][0], det['bbox'][1]],
            [det['bbox'][2], det['bbox'][3]]
        ]], dtype='float32')
        gps_points = cv2.perspectiveTransform(points, matrix)[0]
        transformed_detections.append({
            'id': det['id'],
            'name': det['name'],
            'gps': {'lat': gps_points[0][1], 'lon': gps_points[0][0]},
            'confidence': det['confidence']
        })
    return transformed_detections

# Example points
image_points = [(100, 100), (200, 100), (200, 200), (100, 200)]
gps_points = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]

perspective_matrix = compute_perspective_transform(gps_points, image_points)
