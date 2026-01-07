// ==================== REGISTRY JAVASCRIPT ====================

// DOM Elements
const searchInput = document.getElementById('searchInput');
const addVehicleBtn = document.getElementById('addVehicleBtn');
const addVehicleModal = document.getElementById('addVehicleModal');
const closeModal = document.getElementById('closeModal');
const cancelBtn = document.getElementById('cancelBtn');
const addVehicleForm = document.getElementById('addVehicleForm');
const vehiclesTableBody = document.getElementById('vehiclesTableBody');
const vehicleCount = document.getElementById('vehicleCount');

let allVehicles = [];

// Event Listeners
addVehicleBtn.addEventListener('click', openModal);
closeModal.addEventListener('click', closeAddModal);
cancelBtn.addEventListener('click', closeAddModal);
addVehicleForm.addEventListener('submit', handleAddVehicle);
searchInput.addEventListener('input', debounce(handleSearch, 300));

// Close modal on outside click
addVehicleModal.addEventListener('click', (e) => {
    if (e.target === addVehicleModal) {
        closeAddModal();
    }
});

// Delete buttons
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('delete-btn')) {
        const plateNumber = e.target.dataset.plate;
        handleDeleteVehicle(plateNumber);
    }
});

/**
 * Open add vehicle modal
 */
function openModal() {
    addVehicleModal.classList.add('active');
    addVehicleForm.reset();
}

/**
 * Close add vehicle modal
 */
function closeAddModal() {
    addVehicleModal.classList.remove('active');
}

/**
 * Handle add vehicle form submission
 */
async function handleAddVehicle(e) {
    e.preventDefault();

    const formData = new FormData(addVehicleForm);
    const data = Object.fromEntries(formData.entries());

    try {
        const result = await apiRequest('/api/vehicles', {
            method: 'POST',
            body: JSON.stringify(data)
        });

        if (result.success) {
            showToast('Vehicle added successfully!', 'success');
            closeAddModal();
            await loadVehicles();
        }
    } catch (error) {
        showToast(`Failed to add vehicle: ${error.message}`, 'error');
    }
}

/**
 * Handle delete vehicle
 */
async function handleDeleteVehicle(plateNumber) {
    if (!confirm(`Are you sure you want to delete vehicle ${plateNumber}?`)) {
        return;
    }

    try {
        const result = await apiRequest(`/api/vehicles/${plateNumber}`, {
            method: 'DELETE'
        });

        if (result.success) {
            showToast('Vehicle deleted successfully!', 'success');
            await loadVehicles();
        }
    } catch (error) {
        showToast(`Failed to delete vehicle: ${error.message}`, 'error');
    }
}

/**
 * Load all vehicles
 */
async function loadVehicles() {
    try {
        const data = await apiRequest('/api/vehicles');

        if (data.success) {
            allVehicles = data.vehicles;
            displayVehicles(allVehicles);
        }
    } catch (error) {
        showToast(`Failed to load vehicles: ${error.message}`, 'error');
    }
}

/**
 * Display vehicles in table
 */
function displayVehicles(vehicles) {
    vehiclesTableBody.innerHTML = '';

    if (vehicles.length === 0) {
        vehiclesTableBody.innerHTML = `
            <tr>
                <td colspan="7" style="text-align: center; padding: 2rem; color: var(--secondary-color);">
                    No vehicles found
                </td>
            </tr>
        `;
        vehicleCount.textContent = 'Total: 0 vehicle(s)';
        return;
    }

    vehicles.forEach(vehicle => {
        const row = document.createElement('tr');
        row.dataset.plate = vehicle.plate_number;

        row.innerHTML = `
            <td><strong>${vehicle.plate_number}</strong></td>
            <td>${vehicle.owner_name}</td>
            <td>${vehicle.vehicle_type || '-'}</td>
            <td>${vehicle.color || '-'}</td>
            <td>${vehicle.model || '-'}</td>
            <td>${formatDate(vehicle.created_at)}</td>
            <td>
                <button class="btn btn-sm btn-danger delete-btn" data-plate="${vehicle.plate_number}">
                    Delete
                </button>
            </td>
        `;

        vehiclesTableBody.appendChild(row);
    });

    vehicleCount.textContent = `Total: ${vehicles.length} vehicle(s)`;
}

/**
 * Handle search
 */
function handleSearch() {
    const query = searchInput.value.toLowerCase().trim();

    if (!query) {
        displayVehicles(allVehicles);
        return;
    }

    const filtered = allVehicles.filter(vehicle => {
        return vehicle.plate_number.toLowerCase().includes(query) ||
               vehicle.owner_name.toLowerCase().includes(query) ||
               (vehicle.vehicle_type && vehicle.vehicle_type.toLowerCase().includes(query)) ||
               (vehicle.model && vehicle.model.toLowerCase().includes(query));
    });

    displayVehicles(filtered);
}

// Initial load
loadVehicles();
