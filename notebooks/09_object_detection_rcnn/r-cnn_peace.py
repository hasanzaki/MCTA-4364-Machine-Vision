import torch, torchvision
import cv2
from PIL import Image

from torchvision import models, transforms

num_classes = 3

# Load the same model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

model.load_state_dict(torch.load(r"model_epoch_10.pth"))

model.eval() # inference mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([transforms.ToTensor()])

cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()

    # Convert the frame to a PIL Image
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    input_img = transform(pil_image)
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
        class_name = "peace"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1-10), 
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
        
        cv2.imshow('frame',frame)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
