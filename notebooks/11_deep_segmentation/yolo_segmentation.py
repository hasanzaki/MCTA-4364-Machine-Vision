import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator
from ultralytics.data.augment import LetterBox

s_mode = 1   # 0: semantic, 1: instance

cap = cv2.VideoCapture(0)

model = YOLO("nose_segmentation.pt")
names = model.names


while cap.isOpened():

    success, im0 = cap.read()
    if not success:
        break

    results = model.track(im0, persist=True)[0]
    annotator = Annotator(im0)

    # Only draw mask+bbox if track id exists
    if results.boxes.id is not None:

        # Extract box, trackid, class and masks
        boxes = results.boxes.xyxy.tolist()
        tids = results.boxes.id.int().tolist()
        clss = results.boxes.cls.cpu().tolist()
        masks = results.masks

        # preprocessing of mask before plotting
        img = LetterBox(masks.shape[1:])(
            image=annotator.result())
        im_gpu = (torch.as_tensor
                  (img, dtype=torch.float16,
                   device=masks.data.device)
                  .permute(2, 0, 1).flip(0)
                  .contiguous() / 255)

        # Masks plotting
        annotator.masks(masks.data, colors=[
            colors(x, True)
            for x in (tids if s_mode==1 else clss)],
                        im_gpu=im_gpu)

        # Bounding box plotting
        for b, t, c in zip(boxes, tids, clss):
            annotator.box_label(
                b,
                color=colors(t if s_mode==1 else c, True),
                label=names[c])

    cv2.imshow("Image segmentation", im0)
   
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()