# imports
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# pre_processing
def L_PP_Data(directory,img_size=(64,64)):
    images = []
    labels = []
    
    categories = ['no', 'yes']

    for category in categories:
        path = os.path.join(directory, category)
        num_label = categories.index(category)
        
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img.flatten())
                labels.append(num_label)
    return np.array(images), np.array(labels)   

#prediction
def predict(model, image_path, img_size=(64,64)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("couldn't load image")
        return None
    
    img = cv2.resize(img, img_size)
    img = img.flatten()
    img = img.reshape(1, -1)
    prediction = model.predict(img)
    
    if prediction[0] == 0:
        res= "no tumor detected"
    else:
        res = "tumor detected"
    print("the result is:", res)
    return res
    
#main
if __name__ == "__main__":
    data_path = r"data_path = brain_tumor_dataset"
    X, y = L_PP_Data(data_path, img_size=(64, 64))
    if len(X) == 0:
        print("check path or data")
    else:
        print(f"loaded {len(X)} images")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("acc is:", acc)
        print("classification report:\n", classification_report(y_test, y_pred))
        
#test model
test_img = r"C:\Users\Mahmoud Hany\OneDrive\Desktop\Task MRI\brain_tumor_dataset\no\N17.jpg"
predict(model, test_img, img_size=(64, 64))