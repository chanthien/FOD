# Đây là code để kiểm tính map, precision, recall, thời gian tính toán cho yolov10 trên file test
# code này thì swr dụng trên kaggle sau khi cài yolov10 với train xong
# code này sẽ so sánh giá trị dự đoán với giá trị trong labels để tính
# code này cần cos các tham chiếu là model, file yaml, test_path, lable_path nên chú ý sửa khi chạy
# 9h30 ngày 7/5



import os
import time
from ultralytics import YOLOv10
from collections import defaultdict
import numpy as np
from sklearn.metrics import average_precision_score


def read_label_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [list(map(float, line.strip().split())) for line in lines]


def calculate_iou(box1, box2):
    # Both boxes are in format [x_center, y_center, width, height]
    x1, y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x2, y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x3, y3 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x4, y4 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    # Calculate intersection
    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)

    width_inter = max(0, x_inter2 - x_inter1)
    height_inter = max(0, y_inter2 - y_inter1)
    area_inter = width_inter * height_inter

    # Calculate union
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x4 - x3) * (y4 - y3)
    area_union = area_box1 + area_box2 - area_inter

    iou = area_inter / area_union if area_union > 0 else 0
    return iou


def evaluate_predictions(predictions, ground_truth, iou_threshold=0.5):
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    for img_id in ground_truth:
        gt_boxes = ground_truth[img_id]
        pred_boxes = predictions.get(img_id, [])

        matched = [False] * len(gt_boxes)

        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if int(gt_box[0]) == int(pred_box[0]):  # Same class
                    iou = calculate_iou(gt_box[1:], pred_box[1:5])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                if not matched[best_gt_idx]:
                    true_positives[int(pred_box[0])] += 1
                    matched[best_gt_idx] = True
                else:
                    false_positives[int(pred_box[0])] += 1
            else:
                false_positives[int(pred_box[0])] += 1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if not matched[gt_idx]:
                false_negatives[int(gt_box[0])] += 1

    return true_positives, false_positives, false_negatives


def calculate_metrics(true_positives, false_positives, false_negatives):
    classes = set(true_positives.keys()) | set(false_positives.keys()) | set(false_negatives.keys())
    precisions = {}
    recalls = {}
    f1_scores = {}

    for cls in classes:
        tp = true_positives[cls]
        fp = false_positives[cls]
        fn = false_negatives[cls]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precisions[cls] = precision
        recalls[cls] = recall
        f1_scores[cls] = f1

    mean_precision = np.mean(list(precisions.values()))
    mean_recall = np.mean(list(recalls.values()))
    mean_f1 = np.mean(list(f1_scores.values()))

    return mean_precision, mean_recall, mean_f1, precisions, recalls, f1_scores


# Load the model
model = YOLOv10('/kaggle/working/runs/detect/train3/weights/best.pt')

# Paths
image_folder = '/kaggle/input/fod-yolov9/test/images/'
label_folder = '/kaggle/input/fod-yolov9/test/labels/'

# Get all image files
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

predictions = {}
ground_truth = {}
total_time = 0

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    # Run inference
    start_time = time.time()
    results = model(source=image_path, conf=0.5)  # Lowered confidence threshold
    end_time = time.time()

    # Calculate processing time
    process_time = end_time - start_time
    total_time += process_time

    # Extract predictions
    img_predictions = []
    for r in results:
        for box in r.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            # Convert to center format and normalize
            x_center = ((x1 + x2) / 2) / r.orig_shape[1]
            y_center = ((y1 + y2) / 2) / r.orig_shape[0]
            width = (x2 - x1) / r.orig_shape[1]
            height = (y2 - y1) / r.orig_shape[0]
            img_predictions.append(
                [int(cls.item()), x_center.item(), y_center.item(), width.item(), height.item(), conf.item()])

    predictions[image_file] = img_predictions

    # Read ground truth
    label_file = image_file.rsplit('.', 1)[0] + '.txt'
    label_path = os.path.join(label_folder, label_file)
    if os.path.exists(label_path):
        ground_truth[image_file] = read_label_file(label_path)
    else:
        print(f"Warning: Label file not found for {image_file}")
        ground_truth[image_file] = []

# Evaluate predictions
true_positives, false_positives, false_negatives = evaluate_predictions(predictions, ground_truth)

# Calculate metrics
mean_precision, mean_recall, mean_f1, precisions, recalls, f1_scores = calculate_metrics(true_positives,
                                                                                         false_positives,
                                                                                         false_negatives)

# Calculate mAP
y_true = []
y_scores = []
for img_id in ground_truth:
    gt_boxes = ground_truth[img_id]
    pred_boxes = predictions.get(img_id, [])

    for gt_box in gt_boxes:
        y_true.append(1)
        best_score = 0
        for pred_box in pred_boxes:
            if int(gt_box[0]) == int(pred_box[0]):
                iou = calculate_iou(gt_box[1:], pred_box[1:5])
                if iou >= 0.5:
                    best_score = max(best_score, pred_box[5])
        y_scores.append(best_score)

    for pred_box in pred_boxes:
        if all(calculate_iou(gt_box[1:], pred_box[1:5]) < 0.5 or int(gt_box[0]) != int(pred_box[0]) for gt_box in
               gt_boxes):
            y_true.append(0)
            y_scores.append(pred_box[5])

mAP = average_precision_score(y_true, y_scores)

# Calculate average processing time per image
avg_time_per_image = total_time / len(image_files)

print(f"mAP: {mAP:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean F1 Score: {mean_f1:.4f}")
print(f"Average Processing Time per Image: {avg_time_per_image:.4f} seconds")

print("\nPer-class metrics:")
for cls in precisions:
    print(f"Class {cls}:")
    print(f"  Precision: {precisions[cls]:.4f}")
    print(f"  Recall: {recalls[cls]:.4f}")
    print(f"  F1 Score: {f1_scores[cls]:.4f}")

print("\nDetailed predictions:")
for img_id, preds in predictions.items():
    print(f"\nImage: {img_id}")
    print("Predictions:")
    for pred in preds:
        print(f"  Class: {pred[0]}, Confidence: {pred[5]:.4f}, Box: {pred[1:5]}")
    print("Ground Truth:")
    for gt in ground_truth[img_id]:
        print(f"  Class: {gt[0]}, Box: {gt[1:]}")