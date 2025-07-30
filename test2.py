import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw, ImageTk
import string
import threading
import time
import io

class HandwritingCanvas:
    def __init__(self, model_path=None):
        """
        Initialize the handwriting canvas application
        """
        self.model = None
        self.model_path = model_path
        self.char_mapping = self._create_emnist_mapping()
        
        # Canvas settings
        self.canvas_width = 800
        self.canvas_height = 400
        self.brush_size = 8
        self.brush_color = "black"
        
        # Drawing state
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # Recognition settings
        self.auto_recognize = True
        self.recognition_delay = 1.0  # seconds
        self.last_draw_time = time.time()
        
        # Initialize GUI
        self.setup_gui()
        
        # Load model if provided
        if model_path:
            self.load_model(model_path)
    
    def _create_emnist_mapping(self):
        """Create mapping from model output indices to characters"""
        chars = []
        # Digits 0-9
        chars.extend([str(i) for i in range(10)])
        # Uppercase A-Z
        chars.extend(list(string.ascii_uppercase))
        # Lowercase a-z
        chars.extend(list(string.ascii_lowercase))
        return chars
    
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Handwriting Word Recognition")
        self.root.geometry("1000x700")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Top frame for controls
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Model loading section
        model_frame = ttk.LabelFrame(control_frame, text="Model", padding="5")
        model_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(model_frame, text="Load Model", command=self.load_model_dialog).pack(side=tk.LEFT, padx=(0, 5))
        self.model_status = ttk.Label(model_frame, text="No model loaded", foreground="red")
        self.model_status.pack(side=tk.LEFT)
        
        # Drawing controls
        draw_frame = ttk.LabelFrame(control_frame, text="Drawing", padding="5")
        draw_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(draw_frame, text="Brush Size:").pack(side=tk.LEFT)
        # Brush size scale
        self.brush_scale = ttk.Scale(draw_frame, from_=2, to=20, orient=tk.HORIZONTAL, 
                                   command=self.update_brush_size)
        self.brush_scale.set(float(self.brush_size))  # Ensure float for scale widget
        self.brush_scale.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Button(draw_frame, text="Clear Canvas", command=self.clear_canvas).pack(side=tk.LEFT)
        
        # Recognition controls
        recog_frame = ttk.LabelFrame(control_frame, text="Recognition", padding="5")
        recog_frame.pack(side=tk.LEFT)
        
        self.auto_var = tk.BooleanVar(value=self.auto_recognize)
        ttk.Checkbutton(recog_frame, text="Auto Recognize", variable=self.auto_var,
                       command=self.toggle_auto_recognize).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(recog_frame, text="Recognize Now", command=self.recognize_manual).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(recog_frame, text="Debug View", command=self.show_debug_window).pack(side=tk.LEFT)
        
        # Canvas frame
        canvas_frame = ttk.LabelFrame(main_frame, text="Write Here", padding="5")
        canvas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Drawing canvas
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height,
                              bg="white", cursor="pencil")
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        # Results frame
        result_frame = ttk.LabelFrame(main_frame, text="Recognition Results", padding="10")
        result_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(1, weight=1)
        
        # Recognized word display
        word_frame = ttk.Frame(result_frame)
        word_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(word_frame, text="Recognized Word:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.word_label = ttk.Label(word_frame, text="", font=("Arial", 16, "bold"), 
                                  foreground="blue", background="lightyellow", 
                                  relief="sunken", padding="10")
        self.word_label.pack(fill=tk.X, pady=(5, 0))
        
        # Character details
        ttk.Label(result_frame, text="Character Details:", font=("Arial", 12, "bold")).grid(row=1, column=0, sticky=tk.W)
        
        # Scrollable text widget for character details
        text_frame = ttk.Frame(result_frame)
        text_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.result_text = tk.Text(text_frame, height=15, width=30, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load a model to begin recognition")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken")
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Start auto-recognition thread
        self.recognition_thread = None
        self.running = True
        self.start_auto_recognition_thread()
    
    def load_model_dialog(self):
        """Open file dialog to load model"""
        file_path = filedialog.askopenfilename(
            title="Select your trained model",
            filetypes=[("Keras Model", "*.keras"), ("H5 Model", "*.h5"), ("All files", "*.*")]
        )
        if file_path:
            self.load_model(file_path)
    
    def load_model(self, model_path):
        """Load the trained character recognition model"""
        try:
            self.model = keras.models.load_model(model_path)
            self.model_path = model_path
            self.model_status.config(text="Model loaded ✓", foreground="green")
            self.status_var.set("Model loaded successfully - Start writing!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model_status.config(text="Model load failed ✗", foreground="red")
    
    def update_brush_size(self, value):
        """Update brush size"""
        try:
            self.brush_size = int(float(value))
        except (ValueError, TypeError):
            self.brush_size = 8  # Default fallback
    
    def toggle_auto_recognize(self):
        """Toggle auto recognition"""
        self.auto_recognize = self.auto_var.get()
    
    def start_drawing(self, event):
        """Start drawing on canvas"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        
        # Ensure brush_size is integer
        brush_size = int(self.brush_size) if isinstance(self.brush_size, (int, float)) else 8
        
        # Draw a dot at start position
        self.canvas.create_oval(event.x - brush_size//2, event.y - brush_size//2,
                              event.x + brush_size//2, event.y + brush_size//2,
                              fill=self.brush_color, outline=self.brush_color)
    
    def draw(self, event):
        """Draw on canvas"""
        if self.drawing and self.last_x and self.last_y:
            # Ensure brush_size is integer
            brush_size = int(self.brush_size) if isinstance(self.brush_size, (int, float)) else 8
            
            # Draw line from last position to current position
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                  width=brush_size, fill=self.brush_color,
                                  capstyle=tk.ROUND, smooth=tk.TRUE)
            self.last_x = event.x
            self.last_y = event.y
            self.last_draw_time = time.time()
    
    def stop_drawing(self, event):
        """Stop drawing"""
        self.drawing = False
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.word_label.config(text="")
        self.result_text.delete(1.0, tk.END)
        self.status_var.set("Canvas cleared - Ready for new input")
    
    def canvas_to_image(self):
        """Convert canvas content to PIL Image"""
        # Get canvas dimensions
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        
        # Create PIL image from canvas
        try:
            # Method 1: Using PostScript (more reliable)
            ps = self.canvas.postscript(colormode='gray')
            img = Image.open(io.BytesIO(ps.encode('latin-1')))
            
            # Convert to grayscale and resize
            img = img.convert('L')
            img = img.resize((self.canvas_width, self.canvas_height))
            
        except:
            # Method 2: Create image from canvas content manually
            img = Image.new('L', (self.canvas_width, self.canvas_height), 255)  # White background
            draw = ImageDraw.Draw(img)
            
            # This is a simplified approach - in practice, you'd need to recreate the drawing
            # For now, we'll use a different approach
            pass
        
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Invert if necessary (text should be white on black for processing)
        if np.mean(img_array) > 127:  # If background is light
            img_array = 255 - img_array
        
        return img_array
    
    def get_canvas_image_alternative(self):
        """Alternative method to get canvas image using tkinter's built-in method"""
        # Create a PIL image with white background
        img = Image.new('RGB', (self.canvas_width, self.canvas_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Get all items from canvas
        items = self.canvas.find_all()
        for item in items:
            coords = self.canvas.coords(item)
            item_type = self.canvas.type(item)
            
            if item_type == 'line' and len(coords) >= 4:
                # Draw line
                try:
                    width = int(float(self.canvas.itemcget(item, 'width')))
                except (ValueError, TypeError):
                    width = 8  # Default width
                draw.line(coords, fill='black', width=width)
            elif item_type == 'oval' and len(coords) >= 4:
                # Draw oval/circle
                draw.ellipse(coords, fill='black')
        
        # Convert to grayscale
        img = img.convert('L')
        img_array = np.array(img)
        
        # Keep the image as is (black text on white background)
        # Don't invert colors - EMNIST expects black text on white background
        
        return img_array
    
    def preprocess_image(self, img_array):
        """Preprocess the canvas image for character recognition"""
        # 1. Invert (white background, black text)
        img = 255 - img_array


        # 4. Apply Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # 5. Threshold to binary
        _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

        # 6. Morphological cleanup
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        return img
    
    def segment_characters(self, img):
        """Segment individual characters from the canvas image"""
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return [], []
        
        # Sort contours left to right based on their x-coordinate
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        character_images = []
        bounding_boxes = []
        
        # Calculate statistics for filtering
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            heights = [cv2.boundingRect(c)[3] for c in contours]
            widths = [cv2.boundingRect(c)[2] for c in contours]
            
            # Use median instead of mean for better filtering
            median_area = np.median(areas)
            median_height = np.median(heights)
            median_width = np.median(widths)
            
            # Set thresholds based on median values
            min_area = max(50, median_area * 0.1)
            max_area = median_area * 10
            min_height = max(15, median_height * 0.4)
            max_height = median_height * 2.5
            min_width = max(8, median_width * 0.3)
            max_width = median_width * 3
        else:
            min_area, max_area = 50, 10000
            min_height, max_height = 15, 200
            min_width, max_width = 8, 150
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Apply multiple filters to remove noise
            if (area < min_area or area > max_area or 
                w < min_width or w > max_width or 
                h < min_height or h > max_height):
                continue
            
            # Aspect ratio filter (characters shouldn't be too wide or too narrow)
            aspect_ratio = w / h
            if aspect_ratio > 3.0 or aspect_ratio < 0.1:
                continue
            
            # Add padding around character for better recognition
            padding = max(3, min(w, h) // 4)
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(img.shape[1] - x_pad, w + 2*padding)
            h_pad = min(img.shape[0] - y_pad, h + 2*padding)
            
            # Extract character region
            char_img = img[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            
            # Skip if character is too small after padding
            if char_img.shape[0] < 10 or char_img.shape[1] < 10:
                continue
            

            char_img = cv2.flip(char_img, 1)
            char_img = cv2.rotate(char_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            character_images.append(char_img)
            bounding_boxes.append((x_pad, y_pad, w_pad, h_pad))
        
        return character_images, bounding_boxes
    
    def prepare_character_for_model(self, char_img):
        """Prepare character image for the trained model"""
        # Resize to 28x28 (EMNIST standard)
        char_img_resized = cv2.resize(char_img, (28, 28))
        
        # EMNIST characters are typically centered and have specific orientation
        # Apply additional preprocessing to match EMNIST format
        
        # Ensure proper centering
        # Find the center of mass of the character
        moments = cv2.moments(char_img_resized)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Calculate shift needed to center the character
            shift_x = 14 - cx  # 14 is center of 28x28 image
            shift_y = 14 - cy
            
            # Create transformation matrix for centering
            if abs(shift_x) > 2 or abs(shift_y) > 2:  # Only shift if significantly off-center
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                char_img_resized = cv2.warpAffine(char_img_resized, M, (28, 28))
        
        # Normalize pixel values to match EMNIST training data
        char_img_resized = char_img_resized.astype('float32') / 255.0
        
        # Add batch dimension and channel dimension
        char_img_resized = char_img_resized.reshape(1, 28, 28, 1)
        
        return char_img_resized
    
    def predict_character(self, char_img):
        """Predict single character using the trained model"""
        if self.model is None:
            return '?', 0.0
        
        try:
            # Prepare image for model
            processed_char = self.prepare_character_for_model(char_img)
            
            # Get prediction
            prediction = self.model.predict(processed_char, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Map to character
            if predicted_class < len(self.char_mapping):
                predicted_char = self.char_mapping[predicted_class]
            else:
                predicted_char = '?'
            
            return predicted_char, confidence
        except Exception as e:
            print(f"Error in character prediction: {e}")
            return '?', 0.0
    
    def recognize_word(self):
        """Recognize word from canvas content"""
        if self.model is None:
            self.status_var.set("Please load a model first")
            return
        
        try:
            # Get image from canvas
            img_array = self.get_canvas_image_alternative()
            
            # Check if canvas has content
            if np.max(img_array) == np.min(img_array):  # All pixels same value
                self.word_label.config(text="")
                self.result_text.delete(1.0, tk.END)
                return
            
            # Preprocess image
            processed_img = self.preprocess_image(img_array)
            
            # Segment characters
            char_images, bboxes = self.segment_characters(processed_img)
            
            if not char_images:
                self.word_label.config(text="No characters detected")
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "No clear characters found.\nTry writing larger or clearer.\nMake sure characters are well separated.")
                return
            
            # Predict each character
            predicted_chars = []
            char_details = []
            
            for i, char_img in enumerate(char_images):
                char, confidence = self.predict_character(char_img)
                predicted_chars.append(char)
                char_details.append(f"Char {i+1}: '{char}' (confidence: {confidence:.3f})")
            
            # Combine characters to form word
            recognized_word = ''.join(predicted_chars)
            
            # Update GUI
            self.word_label.config(text=recognized_word)
            
            # Update character details
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Found {len(char_images)} characters:\n\n")
            for detail in char_details:
                self.result_text.insert(tk.END, detail + "\n")
            
            avg_confidence = np.mean([self.predict_character(ci)[1] for ci in char_images])
            self.result_text.insert(tk.END, f"\nAverage confidence: {avg_confidence:.3f}")
            
            # Add tips for better recognition
            if avg_confidence < 0.7:
                self.result_text.insert(tk.END, "\n\nTips for better recognition:")
                self.result_text.insert(tk.END, "\n• Write characters larger")
                self.result_text.insert(tk.END, "\n• Separate characters clearly")
                self.result_text.insert(tk.END, "\n• Use consistent stroke thickness")
                self.result_text.insert(tk.END, "\n• Write more clearly")
            
            self.status_var.set(f"Recognized: '{recognized_word}' ({len(char_images)} chars, avg conf: {avg_confidence:.2f})")
            
        except Exception as e:
            self.status_var.set(f"Recognition error: {str(e)}")
            print(f"Recognition error: {e}")
            import traceback
            traceback.print_exc()
    
    def show_debug_window(self):
        """Show debug window with processing steps"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        try:
            # Get and process image
            img_array = self.get_canvas_image_alternative()
            processed_img = self.preprocess_image(img_array)
            char_images, bboxes = self.segment_characters(processed_img)
            
            if not char_images:
                messagebox.showinfo("Info", "No characters detected")
                return
            
            # Create debug window
            debug_window = tk.Toplevel(self.root)
            debug_window.title("Debug - Processing Steps")
            debug_window.geometry("1200x800")
            
            # Create notebook for tabs
            import tkinter.ttk as ttk
            notebook = ttk.Notebook(debug_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Original image tab
            orig_frame = ttk.Frame(notebook)
            notebook.add(orig_frame, text="Original")
            
            # Convert to display format
            orig_display = Image.fromarray(img_array).convert('RGB')
            orig_display = orig_display.resize((400, 200))
            orig_photo = ImageTk.PhotoImage(orig_display)
            orig_label = tk.Label(orig_frame, image=orig_photo)
            orig_label.image = orig_photo
            orig_label.pack(pady=20)
            
            # Processed image tab
            proc_frame = ttk.Frame(notebook)
            notebook.add(proc_frame, text="Processed")
            
            proc_display = Image.fromarray(processed_img).convert('RGB')
            proc_display = proc_display.resize((400, 200))
            proc_photo = ImageTk.PhotoImage(proc_display)
            proc_label = tk.Label(proc_frame, image=proc_photo)
            proc_label.image = proc_photo
            proc_label.pack(pady=20)
            
            # Character segments tab
            char_frame = ttk.Frame(notebook)
            notebook.add(char_frame, text="Characters")
            
            # Create scrollable frame for characters
            canvas_scroll = tk.Canvas(char_frame)
            scrollbar = ttk.Scrollbar(char_frame, orient="vertical", command=canvas_scroll.yview)
            scrollable_frame = ttk.Frame(canvas_scroll)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
            )
            
            canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas_scroll.configure(yscrollcommand=scrollbar.set)
            
            # Add character images
            for i, char_img in enumerate(char_images):
                char_pred, char_conf = self.predict_character(char_img)
                
                # Resize for display
                char_display = cv2.resize(char_img, (56, 56))  # 2x size for better visibility
                char_pil = Image.fromarray(char_display).convert('RGB')
                char_photo = ImageTk.PhotoImage(char_pil)
                
                char_container = ttk.Frame(scrollable_frame)
                char_container.pack(side=tk.LEFT, padx=10, pady=10)
                
                char_label = tk.Label(char_container, image=char_photo)
                char_label.image = char_photo
                char_label.pack()
                
                info_label = tk.Label(char_container, text=f"'{char_pred}'\n{char_conf:.3f}", 
                                    font=("Arial", 10))
                info_label.pack()
            
            canvas_scroll.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
        except Exception as e:
            messagebox.showerror("Error", f"Debug error: {str(e)}")
            print(f"Debug error: {e}")
            import traceback
            traceback.print_exc()
    
    def recognize_manual(self):
        """Manually trigger recognition"""
        self.recognize_word()
    
    def start_auto_recognition_thread(self):
        """Start background thread for auto recognition"""
        def auto_recognize_loop():
            while self.running:
                if self.auto_recognize and self.model is not None:
                    # Check if enough time has passed since last drawing
                    if time.time() - self.last_draw_time > self.recognition_delay:
                        try:
                            self.root.after(0, self.recognize_word)  # Run in main thread
                        except:
                            pass
                time.sleep(0.5)  # Check every 500ms
        
        self.recognition_thread = threading.Thread(target=auto_recognize_loop, daemon=True)
        self.recognition_thread.start()
    
    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        self.running = False
        if self.recognition_thread:
            self.recognition_thread.join(timeout=1.0)
        self.root.destroy()

# Additional imports needed
import io

def main():
    """Main function to run the application"""
    # You can provide your model path here or load it through the GUI
    model_path = None
    
    app = HandwritingCanvas(model_path)
    app.run()

if __name__ == "__main__":
    main()

# Example of how to use with your existing model
"""
# Usage example:
1. Run the script
2. Click "Load Model" and select your .keras file
3. Start writing on the white canvas
4. The app will automatically recognize your handwriting
5. View results in the right panel

Features:
- Real-time recognition (can be toggled)
- Adjustable brush size
- Clear canvas button
- Character-by-character confidence scores
- Manual recognition trigger
- Status updates

Requirements:
pip install tkinter opencv-python tensorflow pillow numpy
"""