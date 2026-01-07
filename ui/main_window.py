import tkinter as tk
from tkinter import ttk
from ui.dashboard import Dashboard
from ui.registry import RegistryViewer


class MainWindow(tk.Tk):
    """Main application window."""

    def __init__(self, vehicle_service):
        """Initialize main window.

        Args:
            vehicle_service: VehicleService instance
        """
        super().__init__()

        self.vehicle_service = vehicle_service

        self.title("Vehicle Plate Detection & Registration System")
        self.geometry("1200x700")

        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        # Create menu bar
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.quit)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Dashboard tab
        self.dashboard = Dashboard(self.notebook, self.vehicle_service)
        self.notebook.add(self.dashboard, text="Detection Dashboard")

        # Registry tab
        self.registry = RegistryViewer(self.notebook, self.vehicle_service)
        self.notebook.add(self.registry, text="Vehicle Registry")

        # Status bar
        self.status_bar = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def show_about(self):
        """Show about dialog."""
        about_window = tk.Toplevel(self)
        about_window.title("About")
        about_window.geometry("400x250")
        about_window.resizable(False, False)

        content = ttk.Frame(about_window, padding=20)
        content.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(content, text="Vehicle Plate Detection System",
                         font=('Arial', 14, 'bold'))
        title.pack(pady=(0, 10))

        info_text = """
        A license plate detection and recognition system
        using computer vision and deep learning.

        Features:
        - YOLOv8 for plate detection
        - EasyOCR for text recognition
        - Vehicle registration database
        - Detection logging

        Version: 1.0
        """

        info = ttk.Label(content, text=info_text, justify=tk.LEFT)
        info.pack(pady=10)

        ttk.Button(content, text="Close",
                  command=about_window.destroy).pack(pady=(10, 0))

    def set_status(self, message):
        """Update status bar message.

        Args:
            message: Status message to display
        """
        self.status_bar.config(text=message)
