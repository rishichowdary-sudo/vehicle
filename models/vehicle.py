from datetime import datetime


class Vehicle:
    """Represents a registered vehicle."""

    def __init__(self, plate_number, owner_name, vehicle_type=None,
                 color=None, model=None, created_at=None):
        """Initialize a vehicle object.

        Args:
            plate_number: License plate number (unique identifier)
            owner_name: Name of the vehicle owner
            vehicle_type: Type of vehicle (car, motorcycle, truck, etc.)
            color: Vehicle color
            model: Vehicle make/model
            created_at: Registration timestamp
        """
        self.plate_number = plate_number.upper()
        self.owner_name = owner_name
        self.vehicle_type = vehicle_type
        self.color = color
        self.model = model
        self.created_at = created_at or datetime.now()

    def __repr__(self):
        return f"Vehicle(plate={self.plate_number}, owner={self.owner_name})"

    def to_dict(self):
        """Convert vehicle to dictionary.

        Returns:
            Dictionary representation of the vehicle
        """
        return {
            'plate_number': self.plate_number,
            'owner_name': self.owner_name,
            'vehicle_type': self.vehicle_type,
            'color': self.color,
            'model': self.model,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        }
