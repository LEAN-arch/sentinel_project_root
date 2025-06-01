#!/bin/bash
# setup.sh - For Sentinel Health Co-Pilot Python Backend/Development Environments
# This script sets up a Python virtual environment and installs dependencies
# primarily for Tier 2 (Facility Node) and Tier 3 (Cloud Node) Python components,
# as well as for the overall project development and simulation environment.
# Personal Edge Devices (PEDs) and Supervisor Hubs running native applications
# have separate build/setup processes (e.g., Android Studio, embedded toolchains).

echo "Setting up Sentinel Health Co-Pilot Python virtual environment..."
echo "Targeting: Python backend services, Web Dashboards (Streamlit), Development/Simulation."

# --- Configuration ---
# Determine Project Root: Assume script is in sentinel_project_root/test/scripts/
# So, project root is two levels up from the script's directory.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")" # Should be sentinel_project_root

VENV_NAME_PY="${VENV_NAME_PY:-venv_sentinel_py}" # Allow override, simpler default name
VENV_DIR="${PROJECT_ROOT_DIR}/${VENV_NAME_PY}"
# This requirements file should be at the *project root* for Streamlit Cloud compatibility
# and for general clarity if this script is run from project root.
REQUIREMENTS_FILE="${PROJECT_ROOT_DIR}/requirements.txt"
PYTHON_CMD="${PYTHON_CMD:-python3}"

# --- Helper Functions ---
log_info() { echo "[INFO] $(date +'%Y-%m-%d %H:%M:%S'): $1"; }
log_warn() { echo "[WARN] $(date +'%Y-%m-%d %H:%M:%S'): $1"; }
log_error() { echo "[ERROR] $(date +'%Y-%m-%d %H:%M:%S'): $1" >&2; }

exit_on_error() {
    if [ $? -ne 0 ]; then
        log_error "$1"
        log_error "Setup aborted due to a critical error."
        # Attempt to deactivate venv if active (though VIRTUAL_ENV might not be set if creation failed)
        if [ -n "$VIRTUAL_ENV" ]; then
            log_info "Attempting to deactivate virtual environment..."
            deactivate &>/dev/null # Suppress output of deactivate command itself
        fi
        exit 1
    fi
}

# --- Pre-requisite Checks ---
log_info "Using Project Root: ${PROJECT_ROOT_DIR}"
log_info "Checking for Python interpreter: ${PYTHON_CMD}..."
if ! command -v ${PYTHON_CMD} &> /dev/null; then
    log_error "${PYTHON_CMD} command not found. Please install Python 3 (3.8+ recommended)."
    exit 1
fi
if ! ${PYTHON_CMD} -c "import sys; assert sys.version_info >= (3,8), 'Python 3.8+ required'" &> /dev/null; then
    PYTHON_VERSION=$(${PYTHON_CMD} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_warn "Current Python version is ${PYTHON_VERSION}. Python 3.8+ is recommended. Some dependencies might have issues."
fi

log_info "Checking for Python 'venv' module..."
if ! ${PYTHON_CMD} -m venv -h &> /dev/null; then
    log_error "'venv' module not found for ${PYTHON_CMD}. This is usually part of a standard Python installation. Please verify your Python setup (e.g., 'python3-venv' package on Debian/Ubuntu)."
    exit 1
fi

# --- Virtual Environment Setup ---
if [ ! -d "${VENV_DIR}" ]; then
    log_info "Creating Python virtual environment in ${VENV_DIR}..."
    ${PYTHON_CMD} -m venv "${VENV_DIR}"
    exit_on_error "Failed to create virtual environment at '${VENV_DIR}'. Check permissions and Python 'venv' module."
    log_info "Virtual environment created successfully."
else
    log_info "Python virtual environment ${VENV_DIR} already exists. Skipping creation."
fi

# --- Activate Virtual Environment ---
log_info "Activating Python virtual environment: ${VENV_DIR}/bin/activate"
# shellcheck source=/dev/null # Suppress SC1090/SC1091 for dynamic path that might not exist yet in linting
source "${VENV_DIR}/bin/activate"
if [ -z "$VIRTUAL_ENV" ] || [ "$VIRTUAL_ENV" != "${VENV_DIR}" ]; then # Robust check
    log_error "Virtual environment activation failed or pointed to an unexpected location."
    log_error "Expected VIRTUAL_ENV='${VENV_DIR}', but found '${VIRTUAL_ENV:-Not set}'."
    log_error "Please try activating manually: source \"${VENV_DIR}/bin/activate\""
    exit 1
fi
log_info "Virtual environment successfully activated: $VIRTUAL_ENV"

# --- Pip Upgrade and Dependency Installation ---
log_info "Upgrading pip within the virtual environment..."
pip install --upgrade pip
exit_on_error "Failed to upgrade pip."

if [ -f "$REQUIREMENTS_FILE" ]; then
    log_info "Installing Python dependencies from ${REQUIREMENTS_FILE}..."
    # Advise user about potential complexities with geopandas
    if grep -q "geopandas" "$REQUIREMENTS_FILE"; then
        log_info "Note: 'geopandas' found in requirements. This package has system-level C library dependencies (GDAL, GEOS, PROJ). If installation fails, ensure these are installed on your system first. Consult GeoPandas documentation."
    fi
    pip install -r "${REQUIREMENTS_FILE}"
    if [ $? -eq 0 ]; then
        log_info "Python dependencies installed successfully from ${REQUIREMENTS_FILE}."
    else
        log_warn "Some Python dependencies failed to install from ${REQUIREMENTS_FILE}."
        log_warn "Please review the error messages above. Missing system libraries or package conflicts are common causes."
        log_warn "Consult project documentation or specific package installation guides."
        # Not exiting on error here to allow partial setups for some use cases,
        # but a critical project might choose to exit.
    fi
else
    log_warn "${REQUIREMENTS_FILE} not found. Skipping Python dependency installation."
    log_warn "A requirements file (e.g., requirements.txt at project root) is needed with packages like: streamlit, pandas, geopandas, plotly, numpy."
fi

# --- Post-Setup Information & Guidance ---
echo ""
log_info "Python backend/development environment setup for Sentinel Health Co-Pilot is complete."
log_info "Virtual Environment Path: ${VENV_DIR}"
echo ""
log_info "To use this environment for running Python components (e.g., Streamlit Web Dashboards, Backend Services, Development):"
log_info "1. Activate the virtual environment if not already active:"
log_info "   source \"${VENV_DIR}/bin/activate\""
log_info "2. Navigate to the project root: cd \"${PROJECT_ROOT_DIR}\""
log_info "3. Run your application, e.g.:"
log_info "   streamlit run test/app_home.py"
echo ""
log_info "For native Personal Edge Device (PED) and Supervisor Hub applications:"
log_info " - These require separate development environments (e.g., Android Studio)."
log_info " - Edge AI models (.tflite) must be converted and bundled with these native apps."
log_info " - Refer to specific project documentation for PED/Hub native application setup and model deployment."
echo ""
log_info "To deactivate this Python virtual environment later, type: deactivate"
echo ""

# Check for .env example file for user guidance
ENV_EXAMPLE_FILE="${PROJECT_ROOT_DIR}/.env.example"
ACTUAL_ENV_FILE="${PROJECT_ROOT_DIR}/.env"
if [ -f "$ENV_EXAMPLE_FILE" ] && [ ! -f "$ACTUAL_ENV_FILE" ]; then
    log_warn "An example environment file '.env.example' was found."
    log_warn "Please copy it to '.env' (i.e., cp .env.example .env) in the project root (${PROJECT_ROOT_DIR}) and customize it with your actual environment variables (e.g., MAPBOX_ACCESS_TOKEN, database credentials, API keys)."
fi

exit 0
