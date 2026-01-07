import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os


class Dashboard(tk.Frame):
    """Main dashboard for vehicle plate detection system."""

    def __init__(self, parent, vehicle_service):
        """Initialize dashboard.

        Args:
            parent: Parent tkinter window
            vehicle_service: VehicleService instance
        """
        super().__init__(parent)
        self.vehicle_service = vehicle_service
        self.current_image = None
        self.current_image_path = None

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        # Create main container
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Image display
        left_panel = ttk.LabelFrame(main_container, text="Image Preview", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Image canvas
        self.image_label = tk.Label(left_panel, text="No image loaded",
                                    bg='gray', width=60, height=30)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Image controls
        img_controls = ttk.Frame(left_panel)
        img_controls.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(img_controls, text="Load Image",
                  command=self.load_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(img_controls, text="Detect Plate",
                  command=self.detect_plate).pack(side=tk.LEFT, padx=2)
        ttk.Button(img_controls, text="Clear",
                  command=self.clear_image).pack(side=tk.LEFT, padx=2)

        # Right panel - Results and controls
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))

        # Detection results
        results_frame = ttk.LabelFrame(right_panel, text="Detection Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Results text area
        self.results_text = tk.Text(results_frame, height=15, width=40,
                                   wrap=tk.WORD, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Vehicle registration section
        reg_frame = ttk.LabelFrame(right_panel, text="Quick Registration", padding=10)
        reg_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(reg_frame, text="Plate Number:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.plate_entry = ttk.Entry(reg_frame, width=20)
        self.plate_entry.grid(row=0, column=1, pady=2, padx=5)

        ttk.Label(reg_frame, text="Owner Name:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.owner_entry = ttk.Entry(reg_frame, width=20)
        self.owner_entry.grid(row=1, column=1, pady=2, padx=5)

        ttk.Label(reg_frame, text="Vehicle Type:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.type_entry = ttk.Entry(reg_frame, width=20)
        self.type_entry.grid(row=2, column=1, pady=2, padx=5)

        ttk.Button(reg_frame, text="Register Vehicle",
                  command=self.register_vehicle).grid(row=3, column=0, columnspan=2, pady=10)

        # Statistics
        stats_frame = ttk.LabelFrame(right_panel, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X)

        self.stats_label = ttk.Label(stats_frame, text="Loading stats...",
                                     justify=tk.LEFT)
        self.stats_label.pack(fill=tk.X)

        self.update_stats()

    def load_image(self):
        """Load an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Vehicle Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.log_message(f"Loaded: {os.path.basename(file_path)}")

    def display_image(self, image_path):
        """Display an image in the preview panel.

        Args:
            image_path: Path to the image file
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                messagebox.showerror("Error", "Could not load image")
                return

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize to fit display
            h, w = image_rgb.shape[:2]
            max_w, max_h = 600, 500

            scale = min(max_w/w, max_h/h)
            new_w, new_h = int(w*scale), int(h*scale)

            resized = cv2.resize(image_rgb, (new_w, new_h))

            # Convert to PIL and display
            pil_image = Image.fromarray(resized)
            photo = ImageTk.PhotoImage(pil_image)

            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference

            self.current_image = image

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {e}")

    def detect_plate(self):
        """Detect and read license plate from current image."""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        self.log_message("Processing image...", clear=True)

        try:
            # Process image
            results = self.vehicle_service.process_image(self.current_image_path)

            if not results['success']:
                self.log_message(f"Error: {results['error']}")
                return

            detections = results['detections']

            if not detections:
                self.log_message("No plates detected")
                return

            # Display results
            self.log_message(f"Found {len(detections)} plate(s):\n")

            for i, det in enumerate(detections, 1):
                plate_num = det['plate_number']
                det_conf = det['detection_confidence']
                ocr_conf = det['ocr_confidence']
                registered = det['registered']

                self.log_message(f"Plate {i}: {plate_num}")
                self.log_message(f"  Detection: {det_conf:.2%}")
                self.log_message(f"  OCR: {ocr_conf:.2%}")
                self.log_message(f"  Status: {'REGISTERED' if registered else 'UNREGISTERED'}")

                if registered and det['vehicle_info']:
                    info = det['vehicle_info']
                    self.log_message(f"  Owner: {info['owner_name']}")
                    if info['vehicle_type']:
                        self.log_message(f"  Type: {info['vehicle_type']}")

                self.log_message("")

                # Auto-fill registration form if unregistered
                if not registered and plate_num:
                    self.plate_entry.delete(0, tk.END)
                    self.plate_entry.insert(0, plate_num)

            # Draw detections on image
            self.draw_detections(detections)
            self.update_stats()

        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {e}")
            self.log_message(f"Error: {e}")

    def draw_detections(self, detections):
        """Draw bounding boxes on the image.

        Args:
            detections: List of detection dictionaries
        """
        if self.current_image is None:
            return

        # Create copy of image
        annotated = self.current_image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            plate_num = det['plate_number']
            registered = det['registered']

            # Color: green if registered, red if not
            color = (0, 255, 0) if registered else (0, 0, 255)

            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Add label
            label = f"{plate_num} ({'REG' if registered else 'NEW'})"
            cv2.putText(annotated, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display annotated image
        image_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Resize to fit
        h, w = image_rgb.shape[:2]
        max_w, max_h = 600, 500
        scale = min(max_w/w, max_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        resized = cv2.resize(image_rgb, (new_w, new_h))

        # Convert and display
        pil_image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_image)

        self.image_label.config(image=photo)
        self.image_label.image = photo

    def register_vehicle(self):
        """Register a new vehicle."""
        plate = self.plate_entry.get().strip()
        owner = self.owner_entry.get().strip()
        vehicle_type = self.type_entry.get().strip()

        if not plate or not owner:
            messagebox.showwarning("Warning", "Please enter plate number and owner name")
            return

        success = self.vehicle_service.register_vehicle(
            plate_number=plate,
            owner_name=owner,
            vehicle_type=vehicle_type if vehicle_type else None
        )

        if success:
            messagebox.showinfo("Success", f"Vehicle {plate} registered successfully!")
            self.plate_entry.delete(0, tk.END)
            self.owner_entry.delete(0, tk.END)
            self.type_entry.delete(0, tk.END)
            self.update_stats()
        else:
            messagebox.showerror("Error", "Failed to register vehicle (may already exist)")

    def clear_image(self):
        """Clear the current image."""
        self.current_image = None
        self.current_image_path = None
        self.image_label.config(image="", text="No image loaded")
        self.log_message("Image cleared", clear=True)

    def log_message(self, message, clear=False):
        """Add a message to the results text area.

        Args:
            message: Message to display
            clear: Clear existing messages first
        """
        self.results_text.config(state=tk.NORMAL)

        if clear:
            self.results_text.delete(1.0, tk.END)

        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.results_text.config(state=tk.DISABLED)

    def update_stats(self):
        """Update statistics display."""
        try:
            stats = self.vehicle_service.get_stats()
            text = f"Total Vehicles: {stats['total_vehicles']}\n"
            text += f"Recent Detections: {stats['recent_detections']}"
            self.stats_label.config(text=text)
        except Exception as e:
            self.stats_label.config(text=f"Error loading stats: {e}")
