from detecto.core import Model, Dataset
from detecto import core, utils, visualize
dataset = Dataset('120/New folder/')
model = Model([
    'top_left', 'top_right', 'bot_left', 'bot_right'
])
#losses = model.fit(dataset, epochs=50, verbose=True, learning_rate=0.001)

fname = '1.jpg'
image = utils.read_image(fname)
labels, boxes, scores = model.predict(image)
print(boxes[0])

# import cv2
#
# for i, bbox in enumerate(boxes):
#     bbox = list(map(int, bbox))
#     x_min, y_min, x_max, y_max = bbox
#     cv2.rectangle(image,(x_min,y_min),(x_max,y_max),(0,255,0),2)
#     cv2.putText(image, labels[i], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
# cv2.imshow("a",image)
# cv2.waitKey()