"""
Vehicle Plate Detection & Registration System
Main application entry point
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import MainWindow
from services.vehicle_service import VehicleService
from models.database import Database
import config


def main():
    """Main application entry point."""
    print("=" * 60)
    print("Vehicle Plate Detection & Registration System")
    print("=" * 60)
    print()

    try:
        # Initialize database
        print("Initializing database...")
        db = Database(config.DATABASE_PATH)

        # Initialize vehicle service
        print("Loading AI models (this may take a moment)...")
        vehicle_service = VehicleService(
            database=db,
            detector_model=config.YOLO_MODEL_PATH,
            use_gpu=config.USE_GPU
        )

        print()
        print("=" * 60)
        print("System initialized successfully!")
        print("=" * 60)
        print()

        # Create and run GUI
        app = MainWindow(vehicle_service)
        app.mainloop()

    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

    except Exception as e:
        print(f"\nError starting application: {e}")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
