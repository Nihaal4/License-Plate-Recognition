import cv2
import pytesseract
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Configure Tesseract path (change this if you're on Windows)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detection")
        self.root.geometry("800x600")
        self.image_path = None

        # Upload button
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image, bg="#007BFF", fg="white")
        self.upload_btn.pack(pady=10)

        # Image display
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Detect button
        self.detect_btn = tk.Button(root, text="Detect Plate", command=self.detect_plate, bg="green", fg="white")
        self.detect_btn.pack(pady=10)

        # Result text
        self.result_label = tk.Label(root, text="", font=("Arial", 16))
        self.result_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)
            img.thumbnail((500, 400))
            self.tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_img)

    def detect_plate(self):
        if not self.image_path:
            self.result_label.config(text="❌ No image selected.")
            return

        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 100, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        plate_contour = None
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                plate_contour = approx
                break

        if plate_contour is not None:
            x, y, w, h = cv2.boundingRect(plate_contour)
            plate = gray[y:y + h, x:x + w]
            _, thresh = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            plate_number = pytesseract.image_to_string(thresh, config='--psm 8').strip()

            self.result_label.config(text=f"🔍 Detected Plate: {plate_number}")
        else:
            self.result_label.config(text="❌ License plate not detected.")

# Launch the app
if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()
