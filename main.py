import os
import cv2
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

def L_PP_Data(directory, img_size=(64, 64)):
    images = []
    labels = []
    categories = ['no', 'yes']

    for category in categories:
        path = os.path.join(directory, category)
        num_label = categories.index(category)
        
        if not os.path.exists(path):
            continue
            
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img.flatten())
                labels.append(num_label)
                
    return np.array(images), np.array(labels)            

def predict(model, pca, opencv_img, img_size=(64, 64)):
    img = cv2.resize(opencv_img, img_size)
    img = img.flatten()
    img = img.reshape(1, -1)
    
    img_pca = pca.transform(img)
    prediction = model.predict(img_pca)
    
    if prediction[0] == 0:
        res = "no tumor detected"
    else:
        res = "tumor detected"
        
    return res

if __name__ == "__main__":
    st.title("Brain Tumor Classification")
    
    data_path = "brain_tumor_dataset"
    X, y = L_PP_Data(data_path, img_size=(64, 64))
    
    if len(X) == 0:
        st.error("check path or data")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pca = PCA(n_components=0.95, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_pca, y_train)
        
        y_pred = model.predict(X_test_pca)
        acc = accuracy_score(y_test, y_pred)
        
        st.write("accuracy is:", acc)
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            test_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            
            st.image(test_img, caption="Uploaded Image", width=300)
            
            result = predict(model, pca, test_img, img_size=(64, 64))
            st.write("the result is:", result)