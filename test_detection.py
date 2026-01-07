"""
Simple test script for vehicle plate detection
Run this to test the system without the GUI
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.vehicle_service import VehicleService
from models.database import Database


def test_system():
    """Test the vehicle detection system."""
    print("=" * 60)
    print("Vehicle Plate Detection System - Test Script")
    print("=" * 60)
    print()

    try:
        # Initialize system
        print("Initializing system...")
        db = Database('test_vehicles.db')
        service = VehicleService(database=db)

        print("System initialized successfully!")
        print()

        # Test 1: Register some vehicles
        print("Test 1: Registering sample vehicles...")
        print("-" * 60)

        vehicles = [
            ("ABC1234", "John Doe", "Car", "Blue", "Toyota Camry"),
            ("XYZ5678", "Jane Smith", "SUV", "Black", "Honda CR-V"),
            ("LMN9012", "Bob Johnson", "Truck", "Red", "Ford F-150"),
        ]

        for plate, owner, vtype, color, model in vehicles:
            success = service.register_vehicle(plate, owner, vtype, color, model)
            if success:
                print(f"✓ Registered: {plate} - {owner}")
            else:
                print(f"✗ Failed: {plate} (may already exist)")

        print()

        # Test 2: List registered vehicles
        print("Test 2: Listing all registered vehicles...")
        print("-" * 60)

        all_vehicles = service.get_all_vehicles()
        print(f"Total vehicles: {len(all_vehicles)}")
        print()

        for vehicle in all_vehicles:
            print(f"  {vehicle.plate_number:10s} | {vehicle.owner_name:20s} | {vehicle.vehicle_type or 'N/A'}")

        print()

        # Test 3: Search for a specific vehicle
        print("Test 3: Searching for a specific vehicle...")
        print("-" * 60)

        search_plate = "ABC1234"
        vehicle = service.get_vehicle_info(search_plate)

        if vehicle:
            print(f"Found vehicle: {search_plate}")
            print(f"  Owner: {vehicle.owner_name}")
            print(f"  Type: {vehicle.vehicle_type}")
            print(f"  Color: {vehicle.color}")
            print(f"  Model: {vehicle.model}")
            print(f"  Registered: {vehicle.created_at}")
        else:
            print(f"Vehicle {search_plate} not found")

        print()

        # Test 4: Process an image (if provided)
        print("Test 4: Image detection test")
        print("-" * 60)

        image_path = input("Enter path to vehicle image (or press Enter to skip): ").strip()

        if image_path and os.path.exists(image_path):
            print(f"Processing image: {image_path}")
            print()

            results = service.process_image(image_path)

            if results['success']:
                print(f"✓ Detection successful!")
                print(f"  Found {len(results['detections'])} plate(s):")
                print()

                for i, det in enumerate(results['detections'], 1):
                    print(f"  Plate {i}:")
                    print(f"    Number: {det['plate_number']}")
                    print(f"    Detection Confidence: {det['detection_confidence']:.2%}")
                    print(f"    OCR Confidence: {det['ocr_confidence']:.2%}")
                    print(f"    Status: {'REGISTERED' if det['registered'] else 'UNREGISTERED'}")

                    if det['vehicle_info']:
                        print(f"    Owner: {det['vehicle_info']['owner_name']}")
                    print()

            else:
                print(f"✗ Detection failed: {results['error']}")

        else:
            print("No image provided, skipping detection test")

        print()

        # Test 5: Statistics
        print("Test 5: System statistics")
        print("-" * 60)

        stats = service.get_stats()
        print(f"Total registered vehicles: {stats['total_vehicles']}")
        print(f"Recent detections: {stats['recent_detections']}")

        print()
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_system()
