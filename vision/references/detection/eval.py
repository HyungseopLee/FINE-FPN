from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()



'''

python eval.py 2>&1 | tee ./evals/fasterrcnn_resnet50_fpn_coco_baseline.log

'''