import torch, torchvision
import cv2
from PIL import Image

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)

preprocess = weights.transforms()
coco_labels = weights.meta["categories"]

model.eval() # inference mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()

    # Convert the frame to a PIL Image
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    input_img = preprocess(pil_image)
    input_tensor = input_img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor) # forward pass

    output = outputs[0]
    
    bboxes = output['boxes'].cpu().numpy().astype(int)
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()

    for bbox, label, score in zip(bboxes, labels, scores):

        if score < 0.8:
            continue

        x1,y1,x2,y2 = bbox
        class_name = coco_labels[label]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1-10), 
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
        
        cv2.imshow('frame',frame)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
