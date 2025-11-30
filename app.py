from models.history_model import init_db, add_record, get_all_records
init_db()

from flask import Flask, render_template, request, send_file
import cv2
import os
import uuid
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

from utils.predict import predict_tumor
from utils.grad_cam import generate_gradcam

app = Flask(__name__)

label_names = ["glioma", "meningioma", "pituitary", "notumor"]


# ---------------------------------------------------
# HOME
# ---------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------
# PREDICTION ROUTE
# ---------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    # ---- Patient Info from form ----
    patient_name   = request.form.get("patient_name", "").strip()
    patient_age    = request.form.get("patient_age", "").strip()
    patient_gender = request.form.get("patient_gender", "").strip()

    # Save MRI
    filename = f"{uuid.uuid4()}.jpg"
    upload_path = os.path.join("static/uploads", filename)
    os.makedirs("static/uploads", exist_ok=True)
    file.save(upload_path)

    # Read image
    img = cv2.imread(upload_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Ensemble prediction
    prediction, confidence, probs = predict_tumor(img)

    # Generate GradCAM
    heatmap, overlay = generate_gradcam(img)
    os.makedirs("static/gradcam", exist_ok=True)

    heatmap_path = f"static/gradcam/heatmap_{filename}"
    overlay_path = f"static/gradcam/overlay_{filename}"

    cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Save history (after heatmap exists)
    add_record(
        upload_path,
        prediction,
        confidence,
        heatmap_path,
        overlay_path,
        str(probs),
        patient_name,
        patient_age,
        patient_gender
    )

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        probs=probs,
        labels=label_names,
        image_path=upload_path,
        heatmap_path=heatmap_path,
        overlay_path=overlay_path,
        filename=filename,
        patient_name=patient_name,
        patient_age=patient_age,
        patient_gender=patient_gender
    )


# ---------------------------------------------------
# PDF REPORT
# ---------------------------------------------------
@app.route("/download_report")
def download_report():
    image_path    = request.args.get("image")
    prediction    = request.args.get("prediction")
    confidence    = request.args.get("confidence")
    heatmap_path  = request.args.get("heatmap")
    overlay_path  = request.args.get("overlay")
    patient_name  = request.args.get("patient_name", "")
    patient_age   = request.args.get("patient_age", "")
    patient_gender= request.args.get("patient_gender", "")

    # Recompute probabilities (for report)
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, _, probs = predict_tumor(img)
    except Exception as e:
        print("Error recomputing probs for PDF:", e)
        probs = None

    pdf_name = f"Report_{uuid.uuid4()}.pdf"
    c = canvas.Canvas(pdf_name, pagesize=A4)
    pw, ph = A4

    # Header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, ph - 60, "BrainTumorX - MRI Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, ph - 90,  f"Prediction : {prediction}")
    c.drawString(50, ph - 110, f"Confidence : {confidence}%")

    # Patient info
    y_info = ph - 140
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_info, "Patient Information:")
    y_info -= 18
    c.setFont("Helvetica", 11)
    c.drawString(60, y_info, f"Name   : {patient_name}")
    y_info -= 15
    c.drawString(60, y_info, f"Age    : {patient_age}")
    y_info -= 15
    c.drawString(60, y_info, f"Gender : {patient_gender}")
    y_info -= 25

    # Probabilities
    y = y_info
    if probs:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Class-wise Probabilities:")
        y -= 20

        c.setFont("Helvetica", 11)
        for label, p in zip(label_names, probs):
            c.drawString(60, y, f"- {label}: {p * 100:.2f}%")
            y -= 15

    # Images
    img_y = 230
    img_w = 150
    img_h = 150

    try:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(60, img_y + img_h + 15, "Uploaded MRI")
        c.drawImage(ImageReader(image_path), 60, img_y, img_w, img_h)
    except Exception as e:
        print("Error inserting MRI in PDF:", e)

    try:
        if heatmap_path:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(230, img_y + img_h + 15, "Grad-CAM Heatmap")
            c.drawImage(ImageReader(heatmap_path), 230, img_y, img_w, img_h)
    except Exception as e:
        print("Error inserting heatmap in PDF:", e)

    try:
        if overlay_path:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(400, img_y + img_h + 15, "Heatmap Overlay")
            c.drawImage(ImageReader(overlay_path), 400, img_y, img_w, img_h)
    except Exception as e:
        print("Error inserting overlay in PDF:", e)

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 50, "Generated by BrainTumorX — Academic / decision-support use only.")

    c.save()
    return send_file(pdf_name, as_attachment=True)


# ---------------------------------------------------
# HISTORY PAGE
# ---------------------------------------------------
@app.route("/history")
def history():
    records = get_all_records()
    return render_template("history.html", records=records)


# ---------------------------------------------------
# EXTRA
# ---------------------------------------------------
@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


# ---------------------------------------------------
# RUN
# ---------------------------------------------------
if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

