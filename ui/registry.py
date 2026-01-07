import tkinter as tk
from tkinter import ttk, messagebox


class RegistryViewer(tk.Frame):
    """Vehicle registry viewer and management."""

    def __init__(self, parent, vehicle_service):
        """Initialize registry viewer.

        Args:
            parent: Parent tkinter window
            vehicle_service: VehicleService instance
        """
        super().__init__(parent)
        self.vehicle_service = vehicle_service

        self.setup_ui()
        self.load_vehicles()

    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_container = ttk.Frame(self, padding=10)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Top controls
        controls = ttk.Frame(main_container)
        controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(controls, text="Refresh",
                  command=self.load_vehicles).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="Add Vehicle",
                  command=self.show_add_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="Delete Selected",
                  command=self.delete_selected).pack(side=tk.LEFT, padx=2)

        # Search
        ttk.Label(controls, text="Search:").pack(side=tk.LEFT, padx=(20, 5))
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.filter_vehicles)
        ttk.Entry(controls, textvariable=self.search_var, width=20).pack(side=tk.LEFT)

        # Vehicle list (treeview)
        list_frame = ttk.Frame(main_container)
        list_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        vsb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        hsb = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)

        # Treeview
        columns = ('plate', 'owner', 'type', 'color', 'model', 'registered')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings',
                                 yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        # Define columns
        self.tree.heading('plate', text='Plate Number')
        self.tree.heading('owner', text='Owner Name')
        self.tree.heading('type', text='Vehicle Type')
        self.tree.heading('color', text='Color')
        self.tree.heading('model', text='Model')
        self.tree.heading('registered', text='Registered Date')

        self.tree.column('plate', width=120)
        self.tree.column('owner', width=150)
        self.tree.column('type', width=100)
        self.tree.column('color', width=100)
        self.tree.column('model', width=150)
        self.tree.column('registered', width=150)

        self.tree.pack(fill=tk.BOTH, expand=True)

        # Bind double-click to view details
        self.tree.bind('<Double-1>', self.view_details)

        # Bottom info
        info_frame = ttk.Frame(main_container)
        info_frame.pack(fill=tk.X, pady=(10, 0))

        self.info_label = ttk.Label(info_frame, text="0 vehicles registered")
        self.info_label.pack(side=tk.LEFT)

    def load_vehicles(self):
        """Load all vehicles from database."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Load vehicles
        vehicles = self.vehicle_service.get_all_vehicles()

        for vehicle in vehicles:
            self.tree.insert('', tk.END, values=(
                vehicle.plate_number,
                vehicle.owner_name,
                vehicle.vehicle_type or '-',
                vehicle.color or '-',
                vehicle.model or '-',
                vehicle.created_at
            ))

        self.info_label.config(text=f"{len(vehicles)} vehicles registered")

    def filter_vehicles(self, *args):
        """Filter vehicles based on search query."""
        query = self.search_var.get().upper()

        if not query:
            self.load_vehicles()
            return

        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Load and filter vehicles
        vehicles = self.vehicle_service.get_all_vehicles()
        filtered = [v for v in vehicles if query in v.plate_number.upper()
                   or query in v.owner_name.upper()]

        for vehicle in filtered:
            self.tree.insert('', tk.END, values=(
                vehicle.plate_number,
                vehicle.owner_name,
                vehicle.vehicle_type or '-',
                vehicle.color or '-',
                vehicle.model or '-',
                vehicle.created_at
            ))

        self.info_label.config(text=f"{len(filtered)} of {len(vehicles)} vehicles")

    def show_add_dialog(self):
        """Show dialog to add a new vehicle."""
        dialog = tk.Toplevel(self)
        dialog.title("Add Vehicle")
        dialog.geometry("400x300")
        dialog.resizable(False, False)

        content = ttk.Frame(dialog, padding=20)
        content.pack(fill=tk.BOTH, expand=True)

        # Form fields
        ttk.Label(content, text="Plate Number *:").grid(row=0, column=0, sticky=tk.W, pady=5)
        plate_entry = ttk.Entry(content, width=30)
        plate_entry.grid(row=0, column=1, pady=5, padx=5)

        ttk.Label(content, text="Owner Name *:").grid(row=1, column=0, sticky=tk.W, pady=5)
        owner_entry = ttk.Entry(content, width=30)
        owner_entry.grid(row=1, column=1, pady=5, padx=5)

        ttk.Label(content, text="Vehicle Type:").grid(row=2, column=0, sticky=tk.W, pady=5)
        type_entry = ttk.Entry(content, width=30)
        type_entry.grid(row=2, column=1, pady=5, padx=5)

        ttk.Label(content, text="Color:").grid(row=3, column=0, sticky=tk.W, pady=5)
        color_entry = ttk.Entry(content, width=30)
        color_entry.grid(row=3, column=1, pady=5, padx=5)

        ttk.Label(content, text="Model:").grid(row=4, column=0, sticky=tk.W, pady=5)
        model_entry = ttk.Entry(content, width=30)
        model_entry.grid(row=4, column=1, pady=5, padx=5)

        def save_vehicle():
            plate = plate_entry.get().strip()
            owner = owner_entry.get().strip()

            if not plate or not owner:
                messagebox.showwarning("Warning", "Plate number and owner name are required")
                return

            success = self.vehicle_service.register_vehicle(
                plate_number=plate,
                owner_name=owner,
                vehicle_type=type_entry.get().strip() or None,
                color=color_entry.get().strip() or None,
                model=model_entry.get().strip() or None
            )

            if success:
                messagebox.showinfo("Success", f"Vehicle {plate} registered successfully!")
                dialog.destroy()
                self.load_vehicles()
            else:
                messagebox.showerror("Error", "Failed to register vehicle (may already exist)")

        # Buttons
        btn_frame = ttk.Frame(content)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=20)

        ttk.Button(btn_frame, text="Save", command=save_vehicle).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def delete_selected(self):
        """Delete selected vehicle."""
        selection = self.tree.selection()

        if not selection:
            messagebox.showwarning("Warning", "Please select a vehicle to delete")
            return

        item = self.tree.item(selection[0])
        plate_number = item['values'][0]

        if messagebox.askyesno("Confirm Delete",
                              f"Delete vehicle {plate_number}?"):
            success = self.vehicle_service.delete_vehicle(plate_number)

            if success:
                self.load_vehicles()
                messagebox.showinfo("Success", "Vehicle deleted successfully")
            else:
                messagebox.showerror("Error", "Failed to delete vehicle")

    def view_details(self, event):
        """View detailed information about a vehicle."""
        selection = self.tree.selection()

        if not selection:
            return

        item = self.tree.item(selection[0])
        values = item['values']

        details = f"""
Vehicle Details:

Plate Number: {values[0]}
Owner Name: {values[1]}
Vehicle Type: {values[2]}
Color: {values[3]}
Model: {values[4]}
Registered: {values[5]}
        """

        messagebox.showinfo("Vehicle Details", details)
