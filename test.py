import cv2
import numpy as np
import pickle
from keras.models import load_model

# Load label encoder và mô hình
with open('label_encoder.pickle', 'rb') as f:
    label_encoder = pickle.load(f)
class_name = label_encoder.classes_

my_model = load_model("vggmodel.h5")  # Hoặc sử dụng weights-61-0.97.h5

cap = cv2.VideoCapture(0)
confidence_threshold = 0.75

while True:
    ret, image_org = cap.read()
    if not ret:
        continue

    # Tiền xử lý ảnh
    image = cv2.resize(image_org, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Dự đoán
    predict = my_model.predict(image)
    predicted_class = np.argmax(predict[0])
    confidence = np.max(predict[0])

    if confidence >= confidence_threshold:
        class_label = class_name[predicted_class]
        print(f"Predicted: {class_label} (Confidence: {confidence:.2f})")
        cv2.putText(image_org, class_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    else:
        print("Low confidence prediction.")

    cv2.imshow("Picture", image_org)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()