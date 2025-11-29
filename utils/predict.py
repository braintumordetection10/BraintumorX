import cv2
import numpy as np
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_pre

# ----------------------- Load Saved Models -----------------------
ensemble = joblib.load("models/brain_tumor_ensemble_model.pkl")
preproc = joblib.load("models/preproc_and_scalers.pkl")

pca = preproc["pca"]
lda = preproc["lda"]
scalers = preproc["scalers"]

label_names = ["glioma", "meningioma", "pituitary", "notumor"]
IMG_SIZE = 224

# ----------------------- Build GAP CNN Models -----------------------
def build_gap_model(model_class):
    base = model_class(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    gap = GlobalAveragePooling2D()(base.output)
    return Model(inputs=base.input, outputs=gap)

vgg_model = build_gap_model(VGG16)
resnet_model = build_gap_model(ResNet50)
inc_model = build_gap_model(InceptionV3)

# ----------------------- Main Prediction Function -----------------------
def predict_tumor(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0).astype("float32")

    # CNN Features
    vgg_feat = vgg_model.predict(vgg_pre(img.copy()), verbose=0)
    res_feat = resnet_model.predict(resnet_pre(img.copy()), verbose=0)
    inc_feat = inc_model.predict(inc_pre(img.copy()), verbose=0)

    # Apply scalers
    vgg_s = scalers['vgg'].transform(vgg_feat)
    res_s = scalers['res'].transform(res_feat)
    inc_s = scalers['inc'].transform(inc_feat)

    # Combine features
    combined = np.concatenate([vgg_s, res_s, inc_s], axis=1)

    # PCA + LDA
    feat_pca = pca.transform(combined)
    feat_lda = lda.transform(feat_pca)

    # Ensemble Prediction
    proba = ensemble.predict_proba(feat_lda)[0]
    pred_class = np.argmax(proba)
    confidence = round(proba[pred_class] * 100, 2)

    return (
        label_names[pred_class],
        confidence,
        proba.tolist()
    )
