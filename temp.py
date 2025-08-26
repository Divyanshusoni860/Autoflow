import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import torch

class FlowchartToCodeConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Flowchart to Code Converter")
        self.root.geometry("1200x800")
        
        # Model setup
        self.weights_path = 'D:/Autoflow/yolov5/runs/train/flowchart-yolo5/weights/best.pt'
        self.model = None
        self.load_model()
        
        # Initialize Tesseract path (update this to your Tesseract installation path)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # GUI Elements
        self.setup_ui()
        
        # Initialize variables
        self.current_image_path = None
        self.detected_elements = []
        self.generated_code = ""
    
    def load_model(self):
        """Load YOLOv5 model"""
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weights_path)
            self.model.conf = 0.25  # confidence threshold
            self.model.iou = 0.45   # NMS IoU threshold
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Left panel - Image and controls
        left_frame = tk.Frame(self.root, width=600, height=700, bg='lightgray')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image display
        self.image_label = tk.Label(left_frame)
        self.image_label.pack(pady=10)
        
        # Load image button
        load_btn = tk.Button(left_frame, text="Load Flowchart Image", command=self.load_image)
        load_btn.pack(pady=5)
        
        # Process button
        process_btn = tk.Button(left_frame, text="Process Flowchart", command=self.process_flowchart)
        process_btn.pack(pady=5)
        
        # Right panel - Results and code
        right_frame = tk.Frame(self.root, width=600, height=700, bg='lightgray')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Detected elements
        tk.Label(right_frame, text="Detected Elements:").pack(anchor=tk.W)
        self.elements_text = scrolledtext.ScrolledText(right_frame, height=10, width=70)
        self.elements_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Generated code
        tk.Label(right_frame, text="Generated Python Code:").pack(anchor=tk.W)
        self.code_text = scrolledtext.ScrolledText(right_frame, height=20, width=70)
        self.code_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Execute button
        execute_btn = tk.Button(right_frame, text="Execute Code", command=self.execute_code)
        execute_btn.pack(pady=5)
        
        # Output console
        tk.Label(right_frame, text="Execution Output:").pack(anchor=tk.W)
        self.output_text = scrolledtext.ScrolledText(right_frame, height=10, width=70)
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if file_path:
            self.current_image_path = file_path
            self.show_image(file_path)
    
    def show_image(self, image_path):
        """Display the selected image"""
        try:
            img = Image.open(image_path)
            img.thumbnail((600, 600))  # Resize to fit
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def process_flowchart(self):
        """Process the flowchart image and generate code"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            # Load and process image
            img = cv2.imread(self.current_image_path)
            if img is None:
                raise ValueError("Failed to load image")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run object detection
            results = self.model(img_rgb)
            
            # Parse detections and extract text with improved OCR
            self.detected_elements = self.parse_detections(results, img_rgb)
            
            # Generate code
            self.generated_code = self.generate_code_from_flowchart(self.detected_elements)
            
            # Display results
            self.display_results()
            
            # Show image with bounding boxes
            rendered_img = results.render()[0]
            self.show_processed_image(rendered_img)
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
    
    def parse_detections(self, results, image):
        """Parse YOLO detections with improved text extraction"""
        elements = []
        df = results.pandas().xyxy[0]  # Detections as pandas DataFrame
        
        for _, row in df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            class_name = row['name']
            confidence = row['confidence']
            
            # Crop the element from image
            element_img = image[y1:y2, x1:x2]
            
            # Enhanced text extraction
            text = self.extract_text_from_element(element_img)
            
            elements.append({
                'type': class_name,
                'coordinates': (x1, y1, x2, y2),
                'confidence': confidence,
                'text': text.strip() if text else ""
            })
        
        return elements
    
    def extract_text_from_element(self, image):
        """Enhanced OCR with better preprocessing for flowchart elements"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Use pytesseract with custom configuration for code-like text
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789=+-*/<>(){}[]:;,.!?\'" '
            text = pytesseract.image_to_string(cleaned, config=custom_config)
            
            # Clean up the text
            text = ' '.join(text.split())  # Remove extra whitespace
            return text
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return ""
    
    def generate_code_from_flowchart(self, elements):
        """Generate Python code from flowchart elements with proper structure"""
        code_lines = []
        variables = set()
        
        # Process elements in order from top to bottom, left to right
        sorted_elements = sorted(elements, key=lambda x: (x['coordinates'][1], x['coordinates'][0]))
        
        # Track decision branches
        decision_stack = []
        
        for element in sorted_elements:
            elem_type = element['type'].lower()
            text = element['text']
            
            if not text and elem_type in ['process', 'input', 'output', 'decision']:
                continue  # Skip elements with no text
            
            if elem_type == 'start':
                code_lines.append("# Program starts")
            elif elem_type == 'end':
                # Close any open decision blocks
                while decision_stack:
                    code_lines.append(" " * (len(decision_stack) * 4) + "pass")
                    decision_stack.pop()
                code_lines.append("# Program ends")
            elif elem_type == 'process':
                # Handle variable assignment
                if '=' in text:
                    var_name = text.split('=')[0].strip()
                    variables.add(var_name)
                # Add proper indentation for decision blocks
                code_lines.append(" " * (len(decision_stack) * 4) + text)
            elif elem_type == 'input':
                var_name = text.split()[0] if text else "input_value"
                prompt = text[len(var_name):].strip() if text else "Enter value"
                code_lines.append(" " * (len(decision_stack) * 4) + f"{var_name} = input('{prompt}: ')")
                variables.add(var_name)
            elif elem_type == 'output':
                output_text = f"print({text})" if not text.startswith('print(') else text
                code_lines.append(" " * (len(decision_stack) * 4) + output_text)
            elif elem_type == 'decision':
                if not text.startswith('if '):
                    text = f"if {text}:"
                code_lines.append(" " * (len(decision_stack) * 4) + text)
                decision_stack.append(text)
        
        # Add variable initialization if needed
        if variables:
            init_code = ["# Initialize variables"]
            for var in variables:
                init_code.append(f"{var} = None")
            code_lines = init_code + ["\n"] + code_lines
        
        return "\n".join(code_lines)
    
    def display_results(self):
        """Display detected elements and generated code"""
        # Clear previous content
        self.elements_text.delete(1.0, tk.END)
        self.code_text.delete(1.0, tk.END)
        
        # Display detected elements
        for elem in self.detected_elements:
            self.elements_text.insert(tk.END, 
                f"Type: {elem['type']}\nText: {elem['text']}\nConfidence: {elem['confidence']:.2f}\n\n")
        
        # Display generated code
        self.code_text.insert(tk.END, self.generated_code)
    
    def show_processed_image(self, image_array):
        """Show the processed image with bounding boxes"""
        try:
            img = Image.fromarray(image_array)
            img.thumbnail((600, 600))  # Resize to fit
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display processed image: {str(e)}")
    
    def execute_code(self):
        """Execute the generated Python code"""
        if not self.generated_code:
            messagebox.showwarning("Warning", "No code to execute. Process a flowchart first.")
            return
        
        try:
            # Clear previous output
            self.output_text.delete(1.0, tk.END)
            
            # Redirect stdout to our text widget
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                # Create a new dictionary for the execution context
                exec_globals = {}
                
                # Execute the code
                exec(self.generated_code, exec_globals)
            
            # Display the output
            output = f.getvalue()
            self.output_text.insert(tk.END, output if output else "Code executed successfully (no output)")
            
        except Exception as e:
            self.output_text.insert(tk.END, f"Error during execution:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FlowchartToCodeConverter(root)
    root.mainloop()