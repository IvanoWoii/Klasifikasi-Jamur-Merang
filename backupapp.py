from time import sleep
import pandas as pd
import cv2
import numpy as np
import tkinter as tk
import serial
from tkinter import ttk
from PIL import Image, ImageTk
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier

class WebcamApp:
    def __init__(self, window, window_title):
        self.knn_model = None
        self.window = window
        self.window.title(window_title)
        
         # Buat label yang menutupi seluruh area aplikasi dan tempatkan gambar di dalamnya
        self.bg_image = tk.PhotoImage(file="background/bgg.png")  # Gantilah dengan path gambar yang sesuai
        self.bg_label = tk.Label(self.window, image=self.bg_image)
        self.bg_label.place(relwidth=1, relheight=1)
        
        title_label = tk.Label(root, text="Aplikasi Klasifikasi Jamur Merang", font=("Arial", 21, "bold"))
        title_label.pack(pady=10)

        # Frame 1: Video dan Tombol Start/Stop
        self.video_frame = tk.Frame(window, bg="lightgreen", padx=10, pady=10,relief=tk.GROOVE, bd=3)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas = tk.Canvas(self.video_frame, width=320, height=240, bg="white", highlightthickness=1, highlightbackground="black")  # Mengatur ukuran canvas video webcam
        self.canvas.pack()
        
        start_image = tk.PhotoImage(file="icon/start.png")
        self.start_button = ttk.Button(self.video_frame, text="Start", command=self.start, style="Start.TButton", image=start_image, compound=tk.LEFT)
        self.start_button.image = start_image 
        self.start_button.pack(side=tk.LEFT, pady=5)
        
        stop_image = tk.PhotoImage(file="icon/stop.png")
        self.stop_button = ttk.Button(self.video_frame, text="Stop", command=self.stop, style="Stop.TButton", image=stop_image, compound=tk.LEFT)
        self.stop_button.image = stop_image
        self.stop_button.pack(side=tk.RIGHT,pady=5)

        # Frame 2: Canvas Capture dan Tombol Capture
        self.capture_frame = tk.Frame(window, bg="lightgreen", padx=10, pady=10,relief=tk.GROOVE, bd=3)
        self.capture_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.capture_canvas = tk.Canvas(self.capture_frame, width=320, height=240, bg="white", highlightthickness=1, highlightbackground="black")  # Mengatur ukuran canvas capture
        self.capture_canvas.pack()
        capute_image = tk.PhotoImage(file="icon/capture.png")
        self.capture_button = ttk.Button(self.capture_frame, text="Capture", command=self.capture, style="Capture.TButton", image=capute_image, compound=tk.LEFT)
        self.caputre_button = capute_image
        self.capture_button.pack(pady=5)
        
        self.result_frame = tk.LabelFrame(window, text="Hasil Klasifikasi", bg="lightgray", padx=12, pady=12, font=("Arial", 12))
        self.result_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.result_label = tk.Label(self.result_frame, text="Belum ada hasil", font=("Arial", 12), bg="lightgray")
        self.result_label.pack()
        
        # Inisialisasi gaya tombol
        self.style = ttk.Style()
        self.style.configure("Start.TButton", background="green")
        self.style.configure("Stop.TButton", background="red")
        self.style.configure("Capture.TButton", background="blue")

        self.is_capturing = False
        self.is_updating = False
        self.is_capturing_running = False
        self.captured_image = None
        
        self.knn_model = None
        
        # Inisialisasi koneksi serial dengan Arduino
        # self.ser = serial.Serial('COM7', 9600)


    def start(self):
        if not self.is_capturing:
            self.vid = cv2.VideoCapture(0)
            self.is_capturing = True
            self.start_update()
            
    def start_update(self):
        if not self.is_updating:
            self.is_updating = True
            self.update()

    def stop(self):
        if self.is_capturing:
            self.vid.release()
            self.is_capturing = False
            self.is_updating = False
            
    def load_training_data(self, filename):
        try:
            data = pd.read_excel(filename)
            X = data[['correlation', 'homogeneity', 'dissimilarity', 'contrast', 'energy', 'ASM']]
            y = data['kelas'].replace({'A': 0, 'B': 1})

            return X, y
        except Exception as e:
            print(f"Error: {e}")
            return None, None
    
    def train_knn_model(self, X, y):
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X, y)
        return knn_model

    
    def classify_image(self, img_features):
        if self.knn_model:
            predicted_label = self.knn_model.predict([img_features])
            predicted_proba = self.knn_model.predict_proba([img_features])
            
            # arduino_command = 'A' if predicted_label[0] == 0 else 'B'
            # self.ser.write(arduino_command.encode())
            
            return predicted_label[0], predicted_proba[0]
        else:
            return None, None

    def capture(self):
        self.is_capturing_running = False
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.resize(frame, (320, 240))

            if self.is_capturing:
                # Ubah ke citra grayscale
                grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                # Thresholding
                ret, img1 = cv2.threshold(grayscale, 129, 255, cv2.THRESH_BINARY_INV)

                # Dilation dan Erosion
                img1 = cv2.dilate(img1, None, iterations=15)
                img1 = cv2.erode(img1, None, iterations=15)

                # Pisahkan saluran warna BGR
                b, g, r = cv2.split(frame)
                rgba = [b, g, r, img1]
                dst = cv2.merge(rgba, 4)
                
                # Temukan dan gambar kontur objek
                contours, hierarchy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                select = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(select)
                segmented_img = dst[y:y+h, x:x+w]

                # Ubah ke citra grayscale
                gray = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2GRAY)

                # Menghitung GLCM
                distances = [5]  # Jarak antar pixel untuk GLCM
                angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Sudut untuk GLCM
                glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

                # Menghitung properti GLCM
                correlation = graycoprops(glcm, 'correlation').ravel()[0]
                homogeneity = graycoprops(glcm, 'homogeneity').ravel()[0]
                dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()[0]
                contrast = graycoprops(glcm, 'contrast').ravel()[0]
                energy = graycoprops(glcm, 'energy').ravel()[0]
                asm = graycoprops(glcm, 'ASM').ravel()[0]
                
                img_features =[correlation,homogeneity,dissimilarity,contrast,energy,asm]
                

                # Menampilkan hasil di frame 3
                result_text = f'Fitur GLCM:\nCorrelation: {correlation:.4f}\nHomogeneity: {homogeneity:.4f}\nDissimilarity: {dissimilarity:.4f}\nContrast: {contrast:.4f}\nEnergy: {energy:.4f}\nASM: {asm:.4f}\n===================='
                self.result_label.config(text=result_text)
                self.result_label.config(wraplength=300, justify="left", anchor="w")
                
                # Menampilkan nilai-nilai fitur ke terminal
                # print("Image Features:")
                # print(img_features)
                
                # Muat data pelatihan
                X, y = self.load_training_data('fix2_data_training.xlsx')  # Sesuaikan dengan nama file Excel Anda
                
                label_mapping = {
                    0: 'GRADE A',
                    1: 'GRADE B',
                }

                # Latih model KNN jika data pelatihan telah dimuat
                if X is not None and y is not None:
                    self.knn_model = self.train_knn_model(X, y)

                    # Klasifikasikan citra hasil segmentasi menggunakan model KNN
                    predicted_label, predicted_proba = self.classify_image(img_features)


                    if predicted_label is not None:
                        # Menampilkan hasil klasifikasi di frame 3
                        predictect_label_text = label_mapping.get(predicted_label, 'Kelas tidak diketahui')
                        result_text += f'\n\nHasil Klasifikasi:\n=== {predictect_label_text} ===\n'
                        
                        # Menampilkan probabilitas untuk setiap kelas
                        proba_text = f'Probabilitas:\nKELAS A: {predicted_proba[0]:.4f}\nKELAS B: {predicted_proba[1]:.4f}'
                        result_text += proba_text 
                        self.result_label.config(text=result_text, font=("Arial", 12), justify="left", anchor="w", wraplength=300, padx=10, fg="black", bd=5)
                else:
                    # Menampilkan pesan jika data pelatihan tidak dapat dimuat
                    result_text += "\n\nSILAHKAN TEKAN TOMBOL START"
                    self.result_label.config(text='Data pelatihan tidak dapat dimuat. Pastikan file Excel ada dan formatnya benar.')

                
            # Ubah ke objek gambar PIL
            img_pil = Image.fromarray(gray)
            # Menam\pilkan gambar di canvas
            self.captured_image = ImageTk.PhotoImage(image=img_pil)
            self.capture_canvas.create_image(0, 0, image=self.captured_image, anchor=tk.NW)
            self.capture_canvas.image = self.captured_image
            
            
    def update(self):
        if self.is_updating:
            if self.is_capturing:
                ret, frame = self.vid.read()
                if ret:
                    # Resize the frame to 320x240
                    frame = cv2.resize(frame, (320, 240))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                    self.canvas.image = photo
                    
                    # arduino_data = self.ser.readline().decode().strip()
                    # try:
                    #     distance = int(arduino_data)
                    #     if 0 <= distance <=18:
                    #         self.capture()
                            
                            
                    # except ValueError:
                    #     print("ERORR INVALID DATA DARI ARDUINO")
            # Call the update function recursively after a delay
            self.window.after(10, self.update)

    def __del__(self):
        if self.is_capturing:
            self.vid.release()
        self.ser.close()

root = tk.Tk()
root.geometry("1024x768")
app = WebcamApp(root, "APP PYTHON KONVEYOR")
root.mainloop()
