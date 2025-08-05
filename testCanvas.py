import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk, ImageDraw

class HandwritingApp:
    def __init__(self):
        self.model = None
        self.drawing = False
        self.last_x = self.last_y = None
        self.current_image = None
        self.load_default_model()
        self.setup_gui()
    
    def get_character_from_label(self, label):
        """
        Convert EMNIST label to character based on a custom explicit mapping for a subset of lowercase letters,
        while using generalizable ASCII arithmetic for digits and uppercase letters.
        """
        custom_lowercase_map = {
            36: 'a',
            37: 'b',
            38: 'd',
            39: 'e',
            40: 'f',
            41: 'g',
            42: 'h',
            43: 'n',
            44: 'q',
            45: 'r',
            46: 't'
        }

        if 0 <= label <= 9:
            return chr(ord('0') + label)
        elif 10 <= label <= 35:
            return chr(ord('A') + label - 10)
        elif 36 <= label <= 46:
            if label in custom_lowercase_map:
                return custom_lowercase_map[label]
            else:
                return f"UnknownInCustomLower({label})"
        else:
            return f"Unknown({label})"
    
    def load_default_model(self):
        """Load the default model if it exists"""
        import os
        model_path = r"model\emnist_balanced_cnn_best.h5"
        if os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                print(f"Default model loaded from: {model_path}")
            except Exception as e:
                print(f"Failed to load default model: {e}")
                self.model = None
        else:
            print(f"Default model not found at: {model_path}")
            self.model = None
    
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Handwriting Recognition")
        self.root.geometry("1000x600")
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top controls
        controls = ttk.Frame(main_frame)
        controls.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(controls, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=(0, 5))
        self.model_status = ttk.Label(controls, text="Default model" if self.model else "No model", 
                                     foreground="green" if self.model else "red")
        self.model_status.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Button(controls, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls, text="Recognize", command=self.recognize).pack(side=tk.LEFT)
        
        # Content area
        content = ttk.Frame(main_frame)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Left: Canvas
        canvas_frame = ttk.LabelFrame(content, text="Draw or Load Image", padding="5")
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.canvas = tk.Canvas(canvas_frame, width=500, height=300, bg="white")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Right: Results
        result_frame = ttk.LabelFrame(content, text="Recognition Results", padding="10")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        # Combined result
        ttk.Label(result_frame, text="Recognized Word:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.word_frame = ttk.Frame(result_frame)
        self.word_frame.pack(fill=tk.X, pady=5)
        
        self.result_label = ttk.Label(self.word_frame, text="", font=("Arial", 18, "bold"), 
                                    foreground="blue", background="lightyellow")
        self.result_label.pack(side=tk.LEFT)
        
        self.confidence_label = ttk.Label(self.word_frame, text="", font=("Arial", 12), 
                                        foreground="green")
        self.confidence_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Character details
        ttk.Label(result_frame, text="Character Details:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(20, 5))
        
        # Scrollable frame for characters
        canvas_container = tk.Frame(result_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.char_canvas = tk.Canvas(canvas_container, height=300)
        v_scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.char_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_container, orient="horizontal", command=self.char_canvas.xview)
        
        self.scrollable_frame = ttk.Frame(self.char_canvas)
        self.scrollable_frame.bind("<Configure>", 
                                 lambda e: self.char_canvas.configure(scrollregion=self.char_canvas.bbox("all")))
        
        self.char_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.char_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.char_canvas.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
    
    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Model", 
            filetypes=[("Model files", "*.keras *.h5"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.model = tf.keras.models.load_model(file_path)
                self.model_status.config(text="Model loaded ✓", foreground="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            try:
                img = Image.open(file_path).convert('RGB').resize((500, 300))
                self.current_image = np.array(img)
                
                self.canvas.delete("all")
                photo = ImageTk.PhotoImage(img)
                self.canvas.create_image(250, 150, image=photo)
                self.canvas.image = photo
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
        self.canvas.create_oval(event.x - 4, event.y - 4, event.x + 4, event.y + 4,
                               fill="black", outline="black")
    
    def draw(self, event):
        if self.drawing and self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                   width=8, fill="black", capstyle=tk.ROUND)
            self.last_x, self.last_y = event.x, event.y
    
    def stop_draw(self, event):
        self.drawing = False
        self.last_x = self.last_y = None
    
    def clear(self):
        self.canvas.delete("all")
        self.result_label.config(text="")
        self.confidence_label.config(text="")
        self.current_image = None
        # Clear character details
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
    
    def get_canvas_image(self):
        """Convert canvas to image array"""
        if self.current_image is not None:
            return np.array(Image.fromarray(self.current_image).convert('L'))
        else:
            img = Image.new('RGB', (500, 300), 'white')
            draw = ImageDraw.Draw(img)
            
            items = self.canvas.find_all()
            for item in items:
                coords = self.canvas.coords(item)
                item_type = self.canvas.type(item)
                
                if item_type == 'line' and len(coords) >= 4:
                    draw.line(coords, fill='black', width=8)
                elif item_type == 'oval' and len(coords) >= 4:
                    draw.ellipse(coords, fill='black')
            
            return np.array(img.convert('L'))
    
    def preprocess_image(self, img_array):
        """Preprocess image for recognition"""
        if np.mean(img_array) > 127:
            img = 255 - img_array
        else:
            img = img_array.copy()
        
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        return img
    
    def segment_characters(self, img):
        """Extract individual characters"""
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        characters = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            if area < 50 or w < 8 or h < 15 or w/h > 3 or w/h < 0.1:
                continue
            
            padding = max(3, min(w, h) // 4)
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(img.shape[1] - x_pad, w + 2*padding)
            h_pad = min(img.shape[0] - y_pad, h + 2*padding)
            
            char_img = img[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            
            if char_img.shape[0] >= 10 and char_img.shape[1] >= 10:
                characters.append(char_img)
        
        return characters
    
    def prepare_for_model(self, char_img):
        """Prepare character for model prediction"""
        img = cv2.resize(char_img, (28, 28))
        
        moments = cv2.moments(img)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            shift_x, shift_y = 14 - cx, 14 - cy
            
            if abs(shift_x) > 2 or abs(shift_y) > 2:
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                img = cv2.warpAffine(img, M, (28, 28))
        
        img = img.astype('float32') / 255.0
        return img.reshape(1, 28, 28, 1)
    
    def predict_character(self, char_img):
        """Predict character using model"""
        if self.model is None:
            return '?', 0.0
        
        try:
            char_img = cv2.flip(char_img, 1)
            char_img = cv2.rotate(char_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            processed = self.prepare_for_model(char_img)
            prediction = self.model.predict(processed, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction)
            
            predicted_char = self.get_character_from_label(class_idx)
            return predicted_char, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return '?', 0.0
    
    def display_character_results(self, characters, predictions, confidences):
        """Display character images with predictions in a grid"""
        # Clear previous results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Create grid of character results
        row = 0
        col = 0
        max_cols = 3  # 3 characters per row
        
        for i, (char_img, pred_char, conf) in enumerate(zip(characters, predictions, confidences)):
            # Character frame
            char_frame = ttk.LabelFrame(self.scrollable_frame, text=f"Character {i+1}", padding="5")
            char_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            # Character image
            char_display = cv2.resize(char_img, (60, 60))
            char_pil = Image.fromarray(char_display).convert('RGB')
            char_photo = ImageTk.PhotoImage(char_pil)
            
            img_label = tk.Label(char_frame, image=char_photo, relief="sunken")
            img_label.image = char_photo  # Keep reference
            img_label.pack()
            
            # Prediction
            pred_label = tk.Label(char_frame, text=f"'{pred_char}'", 
                                font=("Arial", 16, "bold"), foreground="blue")
            pred_label.pack(pady=2)
            
            # Confidence
            conf_color = "green" if conf > 0.7 else "orange" if conf > 0.4 else "red"
            conf_label = tk.Label(char_frame, text=f"{conf:.3f}", 
                                font=("Arial", 10), foreground=conf_color)
            conf_label.pack()
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # Update scroll region
        self.scrollable_frame.update_idletasks()
        self.char_canvas.configure(scrollregion=self.char_canvas.bbox("all"))
    
    def recognize(self):
        """Main recognition function"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        try:
            img_array = self.get_canvas_image()
            
            if np.max(img_array) == np.min(img_array):
                self.result_label.config(text="No content")
                self.confidence_label.config(text="")
                return
            
            processed_img = self.preprocess_image(img_array)
            characters = self.segment_characters(processed_img)
            
            if not characters:
                self.result_label.config(text="No characters found")
                self.confidence_label.config(text="")
                return
            
            # Predict all characters
            predictions = []
            confidences = []
            
            for char_img in characters:
                char, conf = self.predict_character(char_img)
                predictions.append(char)
                confidences.append(conf)
            
            # Update main result
            result_word = ''.join(predictions)
            avg_confidence = np.mean(confidences)
            
            self.result_label.config(text=result_word)
            conf_color = "green" if avg_confidence > 0.7 else "orange" if avg_confidence > 0.4 else "red"
            self.confidence_label.config(text=f"({avg_confidence:.3f})", foreground=conf_color)
            
            # Display character details
            self.display_character_results(characters, predictions, confidences)
            
        except Exception as e:
            messagebox.showerror("Error", f"Recognition failed: {e}")
    
    def run(self):
        self.root.mainloop()

def main():
    app = HandwritingApp()
    app.run()

if __name__ == "__main__":
    main()