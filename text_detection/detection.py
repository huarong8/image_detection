import cv2
# Load an image
# you can find some images for testing in "Images for Testing"
#frame = cv2.imread("a.jpg")
frame = cv2.imread("b.jpg")

# Load model weights
model = cv2.dnn_TextDetectionModel_DB("./DB_TD500_resnet50.onnx")
#// Post-processing parameters
binThresh = 0.3;
polyThresh = 0.5;
maxCandidates = 200;
unclipRatio = 2.0;
model.setBinaryThreshold(binThresh) \
     .setPolygonThreshold(polyThresh) \
     .setMaxCandidates(maxCandidates) \
     .setUnclipRatio(unclipRatio)
#// Normalization parameters
#scale = 1.0 / 255.0;
#Scalar mean = Scalar(122.67891434, 116.66876762, 104.00698793);
#// The input shape
#Size inputSize = Size(736, 736);
#model.setInputParams(scale, inputSize, mean);

model.setInputScale(1.0 / 255.0)
model.setInputSize(736, 736)
#model.setInputMean(122.67891434, 116.66876762, 104.00698793)
model.setInputMean(122.67891434)
point = model.detect(frame);
#out = model.predict(frame)
#normAssert(self, out, ref)

##// Visualization
image = cv2.polylines(frame, point[0], True, (0, 255, 0), 2);
cv2.imshow("Text Detection", image);
cv2.waitKey(0);
