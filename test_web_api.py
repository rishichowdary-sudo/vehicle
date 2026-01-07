"""
Test script for the web API
Make sure the web server is running (python webapp.py) before running this!
"""

import requests
import json

BASE_URL = 'http://localhost:5000'


def test_api():
    """Test the web API endpoints."""
    print("=" * 60)
    print("Vehicle Plate Detection System - API Test")
    print("=" * 60)
    print()

    # Test 1: Get stats
    print("Test 1: Getting system statistics...")
    print("-" * 60)
    try:
        response = requests.get(f'{BASE_URL}/api/stats')
        data = response.json()
        if data['success']:
            print(f"✓ Total vehicles: {data['stats']['total_vehicles']}")
            print(f"✓ Recent detections: {data['stats']['recent_detections']}")
        else:
            print(f"✗ Failed: {data.get('error')}")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure the web server is running (python webapp.py)")
        return

    print()

    # Test 2: Register a vehicle
    print("Test 2: Registering a test vehicle...")
    print("-" * 60)
    try:
        vehicle_data = {
            'plate_number': 'WEB001',
            'owner_name': 'API Test User',
            'vehicle_type': 'Test Vehicle',
            'color': 'Blue',
            'model': 'Test Model'
        }

        response = requests.post(
            f'{BASE_URL}/api/vehicles',
            headers={'Content-Type': 'application/json'},
            data=json.dumps(vehicle_data)
        )
        data = response.json()

        if data['success']:
            print(f"✓ {data['message']}")
        else:
            print(f"Note: {data.get('error')} (may already exist)")
    except Exception as e:
        print(f"✗ Error: {e}")

    print()

    # Test 3: Get all vehicles
    print("Test 3: Getting all vehicles...")
    print("-" * 60)
    try:
        response = requests.get(f'{BASE_URL}/api/vehicles')
        data = response.json()

        if data['success']:
            print(f"✓ Found {len(data['vehicles'])} vehicle(s):")
            for vehicle in data['vehicles'][:5]:  # Show first 5
                print(f"  - {vehicle['plate_number']}: {vehicle['owner_name']}")
            if len(data['vehicles']) > 5:
                print(f"  ... and {len(data['vehicles']) - 5} more")
        else:
            print(f"✗ Failed: {data.get('error')}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print()

    # Test 4: Get specific vehicle
    print("Test 4: Getting specific vehicle (WEB001)...")
    print("-" * 60)
    try:
        response = requests.get(f'{BASE_URL}/api/vehicles/WEB001')
        data = response.json()

        if data['success']:
            vehicle = data['vehicle']
            print(f"✓ Found vehicle:")
            print(f"  Plate: {vehicle['plate_number']}")
            print(f"  Owner: {vehicle['owner_name']}")
            print(f"  Type: {vehicle.get('vehicle_type', '-')}")
            print(f"  Color: {vehicle.get('color', '-')}")
            print(f"  Model: {vehicle.get('model', '-')}")
        else:
            print(f"✗ Not found: {data.get('error')}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print()

    # Test 5: Get detection history
    print("Test 5: Getting detection history...")
    print("-" * 60)
    try:
        response = requests.get(f'{BASE_URL}/api/detections?limit=5')
        data = response.json()

        if data['success']:
            print(f"✓ Found {len(data['detections'])} recent detection(s):")
            for det in data['detections']:
                conf = int(det['confidence'] * 100)
                print(f"  - {det['plate_number']}: {conf}% ({det['status']})")
        else:
            print(f"✗ Failed: {data.get('error')}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print()

    # Test 6: Upload image (optional)
    print("Test 6: Image detection test")
    print("-" * 60)

    image_path = input("Enter path to vehicle image (or press Enter to skip): ").strip()

    if image_path:
        try:
            with open(image_path, 'rb') as f:
                response = requests.post(
                    f'{BASE_URL}/api/detect',
                    files={'image': f}
                )
                data = response.json()

                if data['success']:
                    print(f"✓ Detection successful!")
                    print(f"  Found {len(data['detections'])} plate(s):")
                    for det in data['detections']:
                        print(f"    - Plate: {det['plate_number']}")
                        print(f"      Detection: {det['detection_confidence']:.2%}")
                        print(f"      OCR: {det['ocr_confidence']:.2%}")
                        print(f"      Status: {'REGISTERED' if det['registered'] else 'UNREGISTERED'}")
                else:
                    print(f"✗ Failed: {data.get('error')}")
        except FileNotFoundError:
            print(f"✗ Image not found: {image_path}")
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print("Skipped image detection test")

    print()
    print("=" * 60)
    print("API tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_api()
