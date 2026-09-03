import os
import sys
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from src.damage_detector import CarDamageDetector
from src.config import IMAGES_DIR

class CarDamageDesktopApp:
    """
    Desktop GUI application for browsing folders, inspecting vehicle damage images,
    and displaying polygon contours and defect counts.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("CarVision AI - Desktop Damage Inspector")
        self.root.geometry("1100x750")
        self.root.configure(bg='#1e1e24')

        self.detector = None
        self.image_files = []
        self.current_idx = 0
        self.current_cv_image = None
        self.photo_image = None

        self._setup_ui()
        self._init_detector()

        # Load default sample images if available
        if os.path.exists(IMAGES_DIR):
            self._load_folder(IMAGES_DIR)

    def _setup_ui(self):
        # Top Header
        header = tk.Frame(self.root, bg='#111116', height=55)
        header.pack(fill=tk.X, side=tk.TOP)

        title = tk.Label(
            header,
            text="🚗 CarVision AI - Vehicle Damage Inspection",
            font=("Segoe UI", 16, "bold"),
            bg='#111116',
            fg='#00e5ff'
        )
        title.pack(side=tk.LEFT, padx=20, pady=12)

        # Main Layout
        main_pane = tk.Frame(self.root, bg='#1e1e24')
        main_pane.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Left Canvas Panel
        self.canvas_frame = tk.Frame(main_pane, bg='#16161a', bd=1, relief=tk.SOLID)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg='#16161a', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_resize)

        # Right Control & Results Sidebar
        sidebar = tk.Frame(main_pane, bg='#24242e', width=320)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        sidebar.pack_propagate(False)

        # Controls
        btn_open = tk.Button(
            sidebar, text="📂 Open Image Folder",
            font=("Segoe UI", 11, "bold"), bg="#00838f", fg="white",
            relief=tk.FLAT, activebackground="#00acc1", command=self._open_folder_dialog
        )
        btn_open.pack(fill=tk.X, padx=15, pady=12)

        # Confidence Slider
        slider_frame = tk.LabelFrame(sidebar, text="Confidence Threshold", bg='#24242e', fg='#e0e0e0', font=("Segoe UI", 9))
        slider_frame.pack(fill=tk.X, padx=15, pady=8)

        self.conf_var = tk.DoubleVar(value=0.40)
        self.slider = ttk.Scale(slider_frame, from_=0.10, to=0.95, variable=self.conf_var, command=self._on_conf_change)
        self.slider.pack(fill=tk.X, padx=10, pady=5)

        self.conf_label = tk.Label(slider_frame, text="Confidence: 40%", bg='#24242e', fg='#00e5ff')
        self.conf_label.pack(pady=(0, 5))

        # Navigation Buttons
        nav_frame = tk.Frame(sidebar, bg='#24242e')
        nav_frame.pack(fill=tk.X, padx=15, pady=10)

        self.btn_prev = tk.Button(
            nav_frame, text="◀ Previous", width=12, bg="#37474f", fg="white",
            relief=tk.FLAT, command=self._show_prev
        )
        self.btn_prev.pack(side=tk.LEFT)

        self.btn_next = tk.Button(
            nav_frame, text="Next ▶", width=12, bg="#37474f", fg="white",
            relief=tk.FLAT, command=self._show_next
        )
        self.btn_next.pack(side=tk.RIGHT)

        self.page_label = tk.Label(sidebar, text="Image 0 / 0", bg='#24242e', fg='#b0bec5', font=("Segoe UI", 9))
        self.page_label.pack(pady=5)

        # Detections Log
        results_frame = tk.LabelFrame(sidebar, text="Detected Damages", bg='#24242e', fg='#e0e0e0', font=("Segoe UI", 10, "bold"))
        results_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(10, 15))

        self.log_text = tk.Text(results_frame, bg='#181820', fg='#eceff1', font=("Consolas", 10), wrap=tk.WORD, relief=tk.FLAT)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _init_detector(self):
        try:
            self.detector = CarDamageDetector()
            self._log("Ready. Model initialized successfully.")
        except Exception as e:
            self._log(f"Error loading model: {e}")

    def _open_folder_dialog(self):
        folder = filedialog.askdirectory(title="Select Folder Containing Vehicle Images")
        if folder:
            self._load_folder(folder)

    def _load_folder(self, folder_path):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(exts)
        ]
        self.current_idx = 0
        if self.image_files:
            self._process_and_show()
        else:
            self._log("No images found in selected folder.")

    def _on_conf_change(self, val):
        conf = float(val)
        self.conf_label.config(text=f"Confidence: {int(conf * 100)}%")
        if self.image_files:
            self._process_and_show()

    def _show_prev(self):
        if not self.image_files:
            return
        self.current_idx = (self.current_idx - 1) % len(self.image_files)
        self._process_and_show()

    def _show_next(self):
        if not self.image_files:
            return
        self.current_idx = (self.current_idx + 1) % len(self.image_files)
        self._process_and_show()

    def _process_and_show(self):
        if not self.image_files or not self.detector:
            return

        img_path = self.image_files[self.current_idx]
        conf = self.conf_var.get()
        self.page_label.config(text=f"Image {self.current_idx + 1} / {len(self.image_files)}\n{os.path.basename(img_path)}")

        annotated, detections, latency, _ = self.detector.process_image_file(img_path, conf=conf, polygon_mode=True)
        self.current_cv_image = annotated

        # Update Log
        self.log_text.delete('1.0', tk.END)
        self.log_text.insert(tk.END, f"File: {os.path.basename(img_path)}\n")
        self.log_text.insert(tk.END, f"Latency: {latency:.1f}ms\n")
        self.log_text.insert(tk.END, f"Total Defects: {len(detections)}\n")
        self.log_text.insert(tk.END, "------------------------\n")
        for i, d in enumerate(detections, 1):
            self.log_text.insert(tk.END, f"[{i}] {d['class_name']}\n    Conf: {d['confidence']:.1%}\n")

        self._render_canvas()

    def _on_resize(self, event):
        self._render_canvas()

    def _render_canvas(self):
        if self.current_cv_image is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 10 or ch <= 10:
            return

        h, w = self.current_cv_image.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

        rgb = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        self.photo_image = ImageTk.PhotoImage(image=Image.fromarray(resized))
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self.photo_image)

    def _log(self, text):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

def main():
    root = tk.Tk()
    app = CarDamageDesktopApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
