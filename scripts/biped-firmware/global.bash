#!/usr/bin/env bash

# Declare global variables
cpu_cores=$(getconf _NPROCESSORS_ONLN)
eclipse_version="4.25.0"
cmake_target_macos="/Applications/CMake.app/Contents/bin/cmake"
cmake_symlink_macos="/usr/local/bin/cmake"
doxygen_target_macos="/Applications/Doxygen.app/Contents/Resources/doxygen"
doxygen_symlink_macos="/usr/local/bin/doxygen"

# Declare project variables
project_name="biped-firmware"
project_dir="$(pwd)/../.."
project_dir_build="$project_dir/build"
project_dir_cmake="$project_dir/cmake"
project_dir_configs="$project_dir/configs"
project_dir_install="$project_dir/install"
project_dir_src="$project_dir/src"
project_dir_libraries="$project_dir_src/libraries"
project_dir_doxygen="$project_dir/docs/doxygen"

# Declare Arduino CLI variables
arduino_cli_name="arduino-cli"
arduino_cli_version_major=0
arduino_cli_version_minor=21
arduino_cli_version_patch=1
arduino_cli_version="$arduino_cli_version_major.$arduino_cli_version_minor.$arduino_cli_version_patch"
arduino_cli_file_linux="arduino-cli_${arduino_cli_version}_Linux_64bit.tar.gz"
arduino_cli_file_macos="arduino-cli_${arduino_cli_version}_macOS_64bit.tar.gz"
arduino_cli_file="$arduino_cli_file_linux"
arduino_cli_url_linux="https://downloads.arduino.cc/arduino-cli/$arduino_cli_file_linux"
arduino_cli_url_macos="https://downloads.arduino.cc/arduino-cli/$arduino_cli_file_macos"
arduino_cli_url="$arduino_cli_url_linux"
arduino_cli_dir_install="/usr/local/bin"

# Declare Arduino ESP32 Core variables
arduino_esp32_core_name="esp32:esp32"
arduino_esp32_core_version_major=2
arduino_esp32_core_version_minor=0
arduino_esp32_core_version_patch=2
arduino_esp32_core_version="$arduino_esp32_core_version_major.$arduino_esp32_core_version_minor.$arduino_esp32_core_version_patch"
arduino_esp32_core_url="https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json"

# Declare Arduino CMake Toolchain variables
arduino_cmake_toolchain_name="arduino-cmake-toolchain"
arduino_cmake_toolchain_version="953b2e63ddf434868bfba60244fb714262fce5f4"
arduino_cmake_toolchain_version_short="953b2e6"
arduino_cmake_toolchain_url="https://github.com/a9183756-gh/Arduino-CMake-Toolchain.git"

# Declare Adafruit BusIO variables
adafruit_busio_name="Adafruit_BusIO"
adafruit_busio_version_major=1
adafruit_busio_version_minor=14
adafruit_busio_version_patch=0
adafruit_busio_version="$adafruit_busio_version_major.$adafruit_busio_version_minor.$adafruit_busio_version_patch"
adafruit_busio_author="adafruit"

# Declare Adafruit GFX Library variables
adafruit_gfx_library_name="Adafruit_GFX_Library"
adafruit_gfx_library_version_major=1
adafruit_gfx_library_version_minor=11
adafruit_gfx_library_version_patch=3
adafruit_gfx_library_version="$adafruit_gfx_library_version_major.$adafruit_gfx_library_version_minor.$adafruit_gfx_library_version_patch"
adafruit_gfx_library_author="adafruit"

# Declare Adafruit MPU6050 variables
adafruit_mpu6050_name="Adafruit_MPU6050"
adafruit_mpu6050_version_major=2
adafruit_mpu6050_version_minor=0
adafruit_mpu6050_version_patch=6
adafruit_mpu6050_version="$adafruit_mpu6050_version_major.$adafruit_mpu6050_version_minor.$adafruit_mpu6050_version_patch"
adafruit_mpu6050_author="adafruit"

# Declare Adafruit NeoPixel variables
adafruit_neopixel_name="Adafruit_NeoPixel"
adafruit_neopixel_version_major=1
adafruit_neopixel_version_minor=10
adafruit_neopixel_version_patch=4
adafruit_neopixel_version="$adafruit_neopixel_version_major.$adafruit_neopixel_version_minor.$adafruit_neopixel_version_patch"
adafruit_neopixel_author="adafruit"

# Declare Adafruit SH110X variables
adafruit_sh110x_name="Adafruit_SH110X"
adafruit_sh110x_version_major=2
adafruit_sh110x_version_minor=1
adafruit_sh110x_version_patch=8
adafruit_sh110x_version="$adafruit_sh110x_version_major.$adafruit_sh110x_version_minor.$adafruit_sh110x_version_patch"
adafruit_sh110x_author="adafruit"

# Declare Adafruit Unified Sensor variables
adafruit_unified_sensor_name="Adafruit_Unified_Sensor"
adafruit_unified_sensor_version_major=1
adafruit_unified_sensor_version_minor=1
adafruit_unified_sensor_version_patch=6
adafruit_unified_sensor_version="$adafruit_unified_sensor_version_major.$adafruit_unified_sensor_version_minor.$adafruit_unified_sensor_version_patch"
adafruit_unified_sensor_author="adafruit"

# Declare Eigen variables
eigen_name="Eigen"
eigen_version_major=0
eigen_version_minor=2
eigen_version_patch=3
eigen_version="$eigen_version_major.$eigen_version_minor.$eigen_version_patch"
eigen_author="hideakitai"

# Declare Kalman Filter Library variables
kalman_filter_library_name="Kalman_Filter_Library"
kalman_filter_library_version_major=1
kalman_filter_library_version_minor=0
kalman_filter_library_version_patch=2
kalman_filter_library_version="$kalman_filter_library_version_major.$kalman_filter_library_version_minor.$kalman_filter_library_version_patch"
kalman_filter_library_author="TKJElectronics"

# Declare MCP23018 variables
mcp23018_name="MCP23018"
mcp23018_version="cc0b968e3deea9c68adecfd1ce99a14d29f3d486"
mcp23018_version_short="cc0b968"
mcp23018_url="https://github.com/maniacbug/Arduino.git"

# Declare STM32duino VL53L4CX variables
stm32duino_vl53l4cx_name="STM32duino_VL53L4CX"
stm32duino_vl53l4cx_version_major=1
stm32duino_vl53l4cx_version_minor=1
stm32duino_vl53l4cx_version_patch=0
stm32duino_vl53l4cx_version="$stm32duino_vl53l4cx_version_major.$stm32duino_vl53l4cx_version_minor.$stm32duino_vl53l4cx_version_patch"
stm32duino_vl53l4cx_author="stm32duino"

function assert_warn()
{
	# Parse arguments
    local status=$1
	local warn_message="$2"

	# Assert status
	if [ $status -ne 0 ]
	then
		# Print warning message
		>&2 echo "[WARN]: $warn_message."
	fi
}

function assert_error()
{
	# Parse arguments
    local status=$1
	local error_message="$2"

	# Assert status
	if [ $status -ne 0 ]
	then
		# Print error message and exit
		>&2 echo "[ERROR]: $error_message. Quit."
		exit $status
	fi
}

function create_symlink()
{
	# Parse arguments
	local target="$1"
	local symlink="$2"

	# Verify symlink
	if [ -f "$symlink" ]
	then
		echo "[INFO]: \"$symlink\" exists. Skip."
		return
	fi

	# Create symlink
	echo "[INFO]: Creating \"$symlink\"..."
	sudo ln -s "$target" "$symlink"
	assert_error $? "Failed to create \"$symlink\""
}

function install_arduino_library()
{
	# Parse arguments
	local name="$1"
	local version="$2"
	local author="$3"
	local config="$4"

	# Create project library directory
	if [ ! -d "$project_dir_libraries" ]
	then
		echo "[INFO]: Creating project library directory..."
		mkdir -p "$project_dir_libraries"
		assert_error $? "Failed to create project library directory"
	fi

	# Verify installation
	if [ -d "$project_dir_libraries/$name" ]
	then
		echo "[INFO]: \"$name\" installation exists. Skip."
		return
	fi

	# Create temporary directory
	local temp_dir
	temp_dir="$(mktemp -d /tmp/$project_name-XXXXXXXXXX)"
	assert_error $? "Failed to create temporary directory"

	# Download library
	echo "[INFO]: Downloading \"$name $version\"..."
	cd "$temp_dir"
	curl -L --progress-bar "https://downloads.arduino.cc/libraries/github.com/$author/$name-$version.zip" -o "$name-$version.zip"
	assert_warn $? "Failed to download \"$name $version\""

	# Install library
	echo "[INFO]: Installing \"$name $version\"..."
	unzip -q "$name-$version.zip"
	mv "$name-$version" "$project_dir_libraries/$name"
	assert_warn $? "Failed to install \"$name $version\""

	# Remove temporary directory
	rm -rf "$temp_dir"
	assert_warn $? "Failed to remove temporary directory"

	# Configure library
	if [ "$config" != "" ]
	then
		"$config"
	fi
}
