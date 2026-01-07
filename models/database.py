import sqlite3
import os
from datetime import datetime
from models.vehicle import Vehicle


class Database:
    """Handles database operations for vehicle registration system."""

    def __init__(self, db_path='vehicle_registry.db'):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create vehicles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT UNIQUE NOT NULL,
                owner_name TEXT NOT NULL,
                vehicle_type TEXT,
                color TEXT,
                model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create detections log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                confidence REAL,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT,
                status TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def add_vehicle(self, plate_number, owner_name, vehicle_type=None,
                   color=None, model=None):
        """Register a new vehicle in the database.

        Args:
            plate_number: License plate number
            owner_name: Name of the owner
            vehicle_type: Type of vehicle
            color: Vehicle color
            model: Vehicle make/model

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO vehicles (plate_number, owner_name, vehicle_type, color, model, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (plate_number.upper(), owner_name, vehicle_type, color, model, datetime.now()))

            conn.commit()
            conn.close()
            return True

        except sqlite3.IntegrityError:
            print(f"Error: Vehicle with plate {plate_number} already exists")
            return False
        except Exception as e:
            print(f"Error adding vehicle: {e}")
            return False

    def get_vehicle_by_plate(self, plate_number):
        """Retrieve vehicle information by plate number.

        Args:
            plate_number: License plate number to search

        Returns:
            Vehicle object if found, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT plate_number, owner_name, vehicle_type, color, model, created_at
                FROM vehicles
                WHERE plate_number = ?
            ''', (plate_number.upper(),))

            result = cursor.fetchone()
            conn.close()

            if result:
                return Vehicle(
                    plate_number=result[0],
                    owner_name=result[1],
                    vehicle_type=result[2],
                    color=result[3],
                    model=result[4],
                    created_at=result[5]
                )
            return None

        except Exception as e:
            print(f"Error retrieving vehicle: {e}")
            return None

    def get_all_vehicles(self):
        """Get all registered vehicles.

        Returns:
            List of Vehicle objects
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT plate_number, owner_name, vehicle_type, color, model, created_at
                FROM vehicles
                ORDER BY created_at DESC
            ''')

            results = cursor.fetchall()
            conn.close()

            vehicles = []
            for row in results:
                vehicle = Vehicle(
                    plate_number=row[0],
                    owner_name=row[1],
                    vehicle_type=row[2],
                    color=row[3],
                    model=row[4],
                    created_at=row[5]
                )
                vehicles.append(vehicle)

            return vehicles

        except Exception as e:
            print(f"Error retrieving vehicles: {e}")
            return []

    def update_vehicle(self, plate_number, **kwargs):
        """Update vehicle information.

        Args:
            plate_number: Plate number to update
            **kwargs: Fields to update (owner_name, vehicle_type, color, model)

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Build update query
            fields = []
            values = []
            for key, value in kwargs.items():
                if key in ['owner_name', 'vehicle_type', 'color', 'model']:
                    fields.append(f"{key} = ?")
                    values.append(value)

            if not fields:
                return False

            values.append(plate_number.upper())
            query = f"UPDATE vehicles SET {', '.join(fields)} WHERE plate_number = ?"

            cursor.execute(query, values)
            conn.commit()
            conn.close()
            return cursor.rowcount > 0

        except Exception as e:
            print(f"Error updating vehicle: {e}")
            return False

    def delete_vehicle(self, plate_number):
        """Delete a vehicle from the database.

        Args:
            plate_number: Plate number to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('DELETE FROM vehicles WHERE plate_number = ?',
                         (plate_number.upper(),))

            conn.commit()
            conn.close()
            return cursor.rowcount > 0

        except Exception as e:
            print(f"Error deleting vehicle: {e}")
            return False

    def log_detection(self, plate_number, confidence, image_path=None, status='unknown'):
        """Log a plate detection event.

        Args:
            plate_number: Detected plate number
            confidence: OCR confidence score
            image_path: Path to the image file
            status: Detection status (registered/unregistered/unknown)

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO detections (plate_number, confidence, image_path, status, detected_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (plate_number.upper(), confidence, image_path, status, datetime.now()))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"Error logging detection: {e}")
            return False

    def get_recent_detections(self, limit=50):
        """Get recent detection logs.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of detection records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT plate_number, confidence, detected_at, image_path, status
                FROM detections
                ORDER BY detected_at DESC
                LIMIT ?
            ''', (limit,))

            results = cursor.fetchall()
            conn.close()

            return results

        except Exception as e:
            print(f"Error retrieving detections: {e}")
            return []

    def get_vehicle_count(self):
        """Get total number of registered vehicles.

        Returns:
            Number of registered vehicles
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM vehicles')
            count = cursor.fetchone()[0]

            conn.close()
            return count

        except Exception as e:
            print(f"Error counting vehicles: {e}")
            return 0
