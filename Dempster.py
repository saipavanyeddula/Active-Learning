import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import numpy as np
from sklearn.cluster import DBSCAN
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from collections import defaultdict

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode
import numpy as np

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    num_forward_passes=10,  # Number of stochastic passes for uncertainty estimation
):

    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Activating Dropout layers for uncertainty estimation in active learning
    for module in model.model.modules():  # Access the underlying model
        if isinstance(module, torch.nn.Dropout):  # Check for Dropout layers
            module.train()  # Keep dropout active during inference
        else:
            module.eval()  # Keep other layers like BatchNorm in eval mode

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        all_preds = []
        for _ in range(num_forward_passes):
            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                if model.xml and im.shape[0] > 1:
                    pred = None
                    for image in ims:
                        if pred is None:
                            pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                        else:
                            pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)),
                                             dim=0)
                    pred = [pred, None]
                else:
                    pred = model(im, augment=augment, visualize=visualize)

            # NMS (Non-Max Suppression)
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                all_preds.append(pred)  # Collect predictions for each forward pass

        # Perform clustering on collected predictions
        clustered_predictions = cluster_predictions(all_preds, eps=0.3, min_samples=7)
        # Process clustered predictions
        uncertainity_preds = []
        valid_preds = []
        uncertainity_threshold = 0.5

        for cluster_id, data in clustered_predictions.items():
            mean_box = np.mean(data['boxes'], axis=0)
            variance_box = np.std(data['boxes'], axis=0)
            mean_conf = np.mean(data['confs'])
            conf_variance = np.var(data['confs'])
            class_id = int(np.round(np.mean(data['class_ids'])))

            # Calculate Dempster-Shafer Theory uncertainty metrics (belief, doubt, ignorance)
            belief, doubt, ignorance, dst_score = dempster_shafer_confidence(data['confs'])

            # Classify as uncertain or valid
            if dst_score < uncertainity_threshold:
                uncertainity_preds.append(np.concatenate((mean_box, [dst_score], [class_id])))
            else:
                valid_preds.append(np.concatenate((mean_box, [dst_score], [class_id])))

        # Convert final predictions to tensors
        uncertainity_preds_tensor = [torch.tensor(uncertainity_preds)] if uncertainity_preds else []
        valid_preds_tensor = [torch.tensor(valid_preds)] if valid_preds else []

        # Output clustered predictions
        print("Uncertainity Predictions:", uncertainity_preds_tensor)
        print("Valid Predictions:", valid_preds_tensor)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"
        # Define the headers
        headers = ["Image Name", "Class Name", "Confidence Score", "x_min", "y_min", "x_max", "y_max"]

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence, bbox_coords):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            bbox_coords = [float(coord) for coord in bbox_coords]  # Ensure bounding box coordinates are floats
            data = {
                "Image Name": image_name,
                "Class Name": prediction,
                "Confidence Score": float(confidence),
                "x_min": bbox_coords[0],
                "y_min": bbox_coords[1],
                "x_max": bbox_coords[2],
                "y_max": bbox_coords[3],
            }
            # Open the file in append mode and write data
            file_exists = csv_path.exists()  # Check if file already exists
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()  # Write headers only if the file is new
                writer.writerow(data)

        # Create directories for saving predictions
        uncertainity_save_dir = save_dir / "Uncertainity_Predictions"
        valid_save_dir = save_dir / "Valid_Predictions"
        images_for_human = save_dir / "Human_to_Annotate"
        uncertainity_save_dir.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist
        valid_save_dir.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't e

        # Process predictions
        for i, det in enumerate([uncertainity_preds_tensor, valid_preds_tensor]):  # per image
            # Set a label for the type of prediction
            pred_type = "Uncertainity" if i == 0 else "Valid"

            for j, det in enumerate(det):  # iterate over each prediction in the current set
                    # print(f"processing predictions")
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f"{i}: "
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                    p = Path(p)  # to Path
                    save_path = str(uncertainity_save_dir / p.name) if pred_type == "Uncertainity" else str(
                        valid_save_dir / p.name)

                    txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
                    s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = names[c] if hide_conf else f"{names[c]}"
                            confidence = float(conf)
                            confidence_str = f"{confidence:.2f}"

                            # Convert bounding box coordinates to a string format
                            bbox_coords = [float(x.item()) for x in xyxy]  # Convert to a list of floats
                            bbox_coords_str = ','.join(f"{coord:.2f}" for coord in bbox_coords)  # Join as a string

                            if save_csv:
                                write_to_csv(p.name, label, confidence_str, bbox_coords)

                            if save_txt:  # Write to file
                                if save_format == 0:
                                    coords = (
                                        (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                    )  # normalized xywh
                                else:
                                    coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()  # xyxy
                                line = (cls, *coords, conf) if save_conf else (cls, *coords)  # label format
                                with open(f"{txt_path}.txt", "a") as f:
                                    f.write(("%g " * len(line)).rstrip() % line + "\n")

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                    # Stream results
                    im0 = annotator.result()
                    if view_img:
                        if platform.system() == "Linux" and p not in windows:
                            windows.append(p)
                            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == "image":
                            # print(f"saving the predictions of image")
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if vid_path[i] != save_path:  # new video
                                vid_path[i] = save_path
                                if isinstance(vid_writer[i], cv2.VideoWriter):
                                    vid_writer[i].release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                            vid_writer[i].write(im0)

          # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


# Define a function for clustering predictions
def cluster_predictions(predictions, eps=0.3, min_samples=8):
    """
    Clusters predictions using DBSCAN based on bounding box coordinates and confidence score.

    Args:
        predictions (list): List of predictions for each forward pass. Each prediction is expected
                            to be a list of tensors in YOLO format: [x1, y1, x2, y2, confidence, class_id].
        eps (float): Maximum distance between points to be considered as in the same cluster.
        min_samples (int): Minimum number of points to form a cluster.

    Returns:
        dict: Dictionary of clusters with bounding boxes and confidence scores.
    """
    all_boxes = []
    for preds in predictions:
        for p in preds:
            if isinstance(p, torch.Tensor):
                p = p.cpu().numpy()  # Convert tensor to NumPy array
            for tensor_data in p:
                boxes = tensor_data[:4]  # Bounding box coordinates
                conf = tensor_data[4]  # Confidence score
                class_id = int(tensor_data[5])  # Class ID
                all_boxes.append(np.concatenate((boxes, [conf, class_id])))

    if len(all_boxes) == 0:
        return {}

    all_boxes = np.array(all_boxes)  # Ensure all_boxes is a NumPy array
    # Normalize bounding box coordinates for clustering
    normalized_boxes = all_boxes[:, :4] / np.max(all_boxes[:, :4], axis=0)
    clustering_features = np.hstack((normalized_boxes, all_boxes[:, 4:5]))  # [x1, y1, x2, y2, confidence]

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(clustering_features)

    # Group predictions by cluster
    clustered_preds = {}
    for i, label in enumerate(cluster_labels):
        if label == -1:
            # Noise points (no cluster)
            continue
        if label not in clustered_preds:
            clustered_preds[label] = {'boxes': [], 'confs': [], 'class_ids': []}
        clustered_preds[label]['boxes'].append(all_boxes[i, :4])
        clustered_preds[label]['confs'].append(all_boxes[i, 4])
        clustered_preds[label]['class_ids'].append(all_boxes[i, 5])

    return clustered_preds


# Dempster-Shafer Theory (DST) Implementation
def dempster_shafer_confidence(confs):
    """
    Computes Dempster-Shafer Theory uncertainty metrics for a list of confidence scores.
    Returns the belief, doubt, and ignorance based on DST.
    """
    belief = np.mean(confs)  # Belief is the mean confidence
    doubt = np.std(confs)  # Doubt is the standard deviation of confidence
    ignorance = max(0, 1 - belief - doubt)  # Ignorance is the remaining uncertainty
    dst_score = belief + doubt
    return belief, doubt, ignorance, dst_score


# Dempster-Shafer Theory (DST) using Bounding Boxes
def dempster_shafer_boxes(boxes):
    """
    Computes Dempster-Shafer Theory uncertainty metrics for a list of bounding boxes.
    Returns the belief, doubt, and ignorance based on DST.
    """
    belief = np.mean(boxes, axis=0)  # Belief is the mean of bounding box coordinates
    doubt = np.std(boxes, axis=0)    # Doubt is the standard deviation of bounding box coordinates
    ignorance = max(0, 1 - (belief + doubt))   # Ignorance is the remaining uncertainty
    dst_score = belief + doubt
    return belief, doubt, ignorance, dst_score


def main(opt):

    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)