#!/bin/bash
# setup.sh - For Sentinel Health Co-Pilot Python Backend/Development Environments
# This script sets up a Python virtual environment and installs dependencies
# primarily for Tier 2 (Facility Node) and Tier 3 (Cloud Node) Python components,
# as well as for the overall project development and simulation environment.
# Personal Edge Devices (PEDs) and Supervisor Hubs running native applications
# have separate build/setup processes (e.g., Android Studio, embedded toolchains).

echo "======================================================================"
echo "Setting up Sentinel Health Co-Pilot Python Virtual Environment..."
echo "Target: Python backend services, Web Dashboards, Dev/Simulation."
echo "======================================================================"
echo

# --- Configuration ---
# Determine Project Root: Assume script is in sentinel_project_root/test/scripts/
# So, project root is two levels up from the script's directory.
SCRIPT_DIR_SETUP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT_DIR_SETUP="$(dirname "$(dirname "$SCRIPT_DIR_SETUP")")" # Should resolve to sentinel_project_root

VENV_NAME_CONFIG_PY="${VENV_NAME_PY:-.venv_sentinel}" # Common practice to prefix venv dir with '.'
VENV_DIR_PATH="${PROJECT_ROOT_DIR_SETUP}/${VENV_NAME_CONFIG_PY}"
# requirements.txt should be at the project root for best practice and compatibility.
REQUIREMENTS_FILE_PATH="${PROJECT_ROOT_DIR_SETUP}/requirements.txt"
PYTHON_EXECUTABLE="${PYTHON_CMD:-python3}" # Allow override, e.g., python3.9

# --- Helper Functions ---
log_info() { echo "[INFO] $(date +'%Y-%m-%d %H:%M:%S') - $1"; }
log_warn() { echo "[WARN] $(date +'%Y-%m-%d %H:%M:%S') - $1"; }
log_error() { echo "[ERROR] $(date +'%Y-%m-%d %H:%M:%S') - $1" >&2; }

exit_on_critical_error() {
    if [ $? -ne 0 ]; then
        log_error "Failed step: $1"
        log_error "Setup aborted due to a critical error."
        # Attempt to deactivate venv if active
        if [ -n "$VIRTUAL_ENV" ]; then
            log_info "Attempting to deactivate virtual environment..."
            deactivate &>/dev/null
        fi
        exit 1
    fi
}

# --- Pre-requisite Checks ---
log_info "Using Project Root: ${PROJECT_ROOT_DIR_SETUP}"
log_info "Checking for Python interpreter: ${PYTHON_EXECUTABLE}..."
if ! command -v ${PYTHON_EXECUTABLE} &> /dev/null; then
    log_error "${PYTHON_EXECUTABLE} command not found. Please install Python 3 (3.8+ recommended)."
    exit 1
fi
PYTHON_VERSION_INFO=$(${PYTHON_EXECUTABLE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
log_info "Found Python version: ${PYTHON_VERSION_INFO}"
if ! ${PYTHON_EXECUTABLE} -c "import sys; assert sys.version_info >= (3,8), 'Python 3.8+ required'" &> /dev/null; then
    log_warn "Current Python version ${PYTHON_VERSION_INFO} is older than recommended 3.8+. Some dependencies might encounter issues."
fi

log_info "Checking for Python 'venv' module..."
if ! ${PYTHON_EXECUTABLE} -m venv -h &> /dev/null; then
    log_error "'venv' module not found for ${PYTHON_EXECUTABLE}. This is usually part of a standard Python installation. Please verify your Python setup (e.g., ensure 'python3-venv' package is installed on Debian/Ubuntu or similar for your OS)."
    exit 1
fi
log_info "'venv' module found."

# --- Virtual Environment Setup ---
if [ ! -d "${VENV_DIR_PATH}" ]; then
    log_info "Creating Python virtual environment in ${VENV_DIR_PATH}..."
    ${PYTHON_EXECUTABLE} -m venv "${VENV_DIR_PATH}"
    exit_on_critical_error "Failed to create virtual environment at '${VENV_DIR_PATH}'. Check permissions and Python 'venv' module status."
    log_info "Virtual environment created successfully."
else
    log_info "Python virtual environment ${VENV_DIR_PATH} already exists. Skipping creation."
fi

# --- Activate Virtual Environment ---
log_info "Activating Python virtual environment: source \"${VENV_DIR_PATH}/bin/activate\""
# shellcheck source=/dev/null # Suppress SC1090/SC1091 for dynamic path
if ! source "${VENV_DIR_PATH}/bin/activate"; then
    log_error "Failed to activate the virtual environment. Attempting to source directly from script failed."
    log_error "Please try activating manually in your shell: source \"${VENV_DIR_PATH}/bin/activate\""
    exit 1
fi

# Robust check for activation
if [ -z "$VIRTUAL_ENV" ] || [ "$VIRTUAL_ENV" != "$(readlink -f "${VENV_DIR_PATH}")" ]; then # Use readlink for canonical path
    log_error "Virtual environment activation failed or pointed to an unexpected location."
    log_error "Expected VIRTUAL_ENV='$(readlink -f "${VENV_DIR_PATH}")', but found '${VIRTUAL_ENV:-Not set}'."
    log_error "Please try activating manually: source \"${VENV_DIR_PATH}/bin/activate\""
    exit 1
fi
log_info "Virtual environment successfully activated: $VIRTUAL_ENV"

# --- Pip Upgrade and Dependency Installation ---
log_info "Upgrading pip within the virtual environment..."
pip install --upgrade pip
exit_on_critical_error "Failed to upgrade pip."

if [ -f "$REQUIREMENTS_FILE_PATH" ]; then
    log_info "Installing Python dependencies from ${REQUIREMENTS_FILE_PATH}..."
    if grep -qE "geopandas|fiona|pyproj|shapely|gdal" "$REQUIREMENTS_FILE_PATH"; then # Broader check for common geo libraries
        log_info "----------------------------------------------------------------------"
        log_info "NOTE: Geospatial libraries (like geopandas, fiona, gdal) found in requirements."
        log_info "These packages often have system-level C library dependencies (e.g., GDAL, GEOS, PROJ)."
        log_info "If installation fails, ensure these are installed on your system first."
        log_info "Consult GeoPandas documentation or search for OS-specific installation guides for these libraries."
        log_info "For example, on Debian/Ubuntu: sudo apt-get install libgdal-dev gdal-bin python3-gdal"
        log_info "----------------------------------------------------------------------"
    fi
    
    pip install -r "${REQUIREMENTS_FILE_PATH}"
    if [ $? -eq 0 ]; then
        log_info "Python dependencies installed successfully from ${REQUIREMENTS_FILE_PATH}."
    else
        log_warn "Some Python dependencies may have failed to install from ${REQUIREMENTS_FILE_PATH}."
        log_warn "Please review the error messages above carefully."
        log_warn "Common causes include missing system libraries (especially for geospatial packages) or package version conflicts."
        log_warn "Consult project documentation or specific package installation guides for troubleshooting."
        log_warn "For a development setup, you may proceed, but the application might not be fully functional."
        # For a production/CI setup, you might want to change this to exit_on_critical_error.
    fi
else
    log_warn "Requirements file '${REQUIREMENTS_FILE_PATH}' not found. Skipping Python dependency installation."
    log_warn "A 'requirements.txt' file at the project root is essential and should list packages like: streamlit, pandas, geopandas, plotly, numpy, etc."
fi

# --- Post-Setup Information & Guidance ---
echo
log_info "=========================================================================="
log_info "Python Environment Setup for Sentinel Health Co-Pilot is Complete!"
log_info "=========================================================================="
log_info "Virtual Environment Path: ${VENV_DIR_PATH}"
echo
log_info "To use this environment:"
log_info "1. Ensure the virtual environment is active:"
log_info "   source \"${VENV_DIR_PATH}/bin/activate\""
log_info "   (Your prompt should change to indicate the active venv, e.g., '(${VENV_NAME_CONFIG_PY}) user@host:...$')"
log_info "2. Navigate to the project root directory:"
log_info "   cd \"${PROJECT_ROOT_DIR_SETUP}\""
log_info "3. Run the Streamlit application (example):"
log_info "   streamlit run test/app_home.py"
echo
log_info "Native PED/Hub Applications:"
log_info " - These require separate development environments (e.g., Android Studio for Android PEDs)."
log_info " - Edge AI models (.tflite) must be converted and bundled with these native apps separately."
log_info " - Refer to specific project documentation for PED/Hub native app setup."
echo
log_info "To deactivate this Python virtual environment later, simply type: deactivate"
echo

# Check for .env.example file for user guidance
ENV_EXAMPLE_FILE_PATH="${PROJECT_ROOT_DIR_SETUP}/.env.example"
ACTUAL_ENV_FILE_PATH="${PROJECT_ROOT_DIR_SETUP}/.env"
if [ -f "$ENV_EXAMPLE_FILE_PATH" ] && [ ! -f "$ACTUAL_ENV_FILE_PATH" ]; then
    log_warn "----------------------------------------------------------------------"
    log_warn "IMPORTANT: An example environment file '.env.example' was found."
    log_warn "Please copy it to '.env' in the project root and customize it:"
    log_warn "   cp \"${ENV_EXAMPLE_FILE_PATH}\" \"${ACTUAL_ENV_FILE_PATH}\""
    log_warn "Then, edit '${ACTUAL_ENV_FILE_PATH}' with your actual environment variables"
    log_warn "(e.g., MAPBOX_ACCESS_TOKEN, database credentials, API keys)."
    log_warn "----------------------------------------------------------------------"
fi

exit 0
