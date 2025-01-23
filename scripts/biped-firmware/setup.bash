#!/usr/bin/env bash

# Go to script directory
cd "$(dirname $0)"

# Source dependencies
source "global.bash"

function validate_os()
{
	# Validate operating system and perform OS-specific tasks
	if [[ "$OSTYPE" == "linux"* ]]
	then
		# Set Arduino CLI variables
		arduino_cli_file="$arduino_cli_file_linux"
		arduino_cli_url="$arduino_cli_url_linux"

		echo "[INFO]: Setting up \"$project_name\" for Linux..."
	elif [[ "$OSTYPE" == "darwin"* ]]
	then
		# Set Arduino CLI variables
		arduino_cli_file="$arduino_cli_file_macos"
		arduino_cli_url="$arduino_cli_url_macos"

		echo "[INFO]: Setting up \"$project_name\" for macOS..."

		# Create symlinks
		create_symlink "$cmake_target_macos" "$cmake_symlink_macos"
		create_symlink "$doxygen_target_macos" "$doxygen_symlink_macos"
	else
		assert_error 1 "Unsupported operating system"
		exit
	fi
}

function install_arduino_cli()
{
	# Verify installation
	if [ -f "$arduino_cli_dir_install/$arduino_cli_name" ]
	then
		echo "[INFO]: \"$arduino_cli_name\" installation exists. Skip."
		return
	fi

	# Create temporary directory
	local temp_dir
	temp_dir="$(mktemp -d /tmp/$project_name-XXXXXXXXXX)"
	assert_error $? "Failed to create temporary directory"

	# Download Arduino CLI
	echo "[INFO]: Downloading \"$arduino_cli_name $arduino_cli_version\"..."
	cd "$temp_dir"
	curl -L --progress-bar "$arduino_cli_url" -o "$arduino_cli_file"
	assert_warn $? "Failed to download \"$arduino_cli_name $arduino_cli_version\""

	# Install Arduino CLI
	echo "[INFO]: Installing \"$arduino_cli_name $arduino_cli_version\"..."
	tar -xzf "$arduino_cli_file"
	sudo cp "$arduino_cli_name" "$arduino_cli_dir_install/$arduino_cli_name"
	assert_warn $? "Failed to install \"$arduino_cli_name $arduino_cli_version\""
	echo "[INFO]: Finished installing \"$arduino_cli_name $arduino_cli_version\"."

	# Remove temporary directory
	rm -rf "$temp_dir"
	assert_warn $? "Failed to remove temporary directory"
}

function install_arduino_esp32_core()
{
	# Verify installation
	if [ $("$arduino_cli_name" core list | grep -ci "$arduino_esp32_core_name $arduino_esp32_core_version") -ne 0 ]
	then
		echo "[INFO]: \"$arduino_esp32_core_name\" installation exists. Skip."
		return
	fi

	# Install Arduino ESP32 Core
	echo "[INFO]: Installing \"$arduino_esp32_core_name $arduino_esp32_core_version\"..."
	"$arduino_cli_name" core update-index --additional-urls "$arduino_esp32_core_url"
	"$arduino_cli_name" core install "$arduino_esp32_core_name@$arduino_esp32_core_version" --additional-urls "$arduino_esp32_core_url"
	assert_error $? "Failed to install \"$arduino_esp32_core_name $arduino_esp32_core_version\""
	echo "[INFO]: Finished installing \"$arduino_esp32_core_name $arduino_esp32_core_version\"."

	# Configure Arduino ESP32 Core
	echo "[INFO]: Configuring \"$arduino_esp32_core_name $arduino_esp32_core_version\"..."
	cd "$($arduino_cli_name core list -v | grep -i 'Loading package esp32' | awk 'NF>1{print $NF}' -)/esp32/$arduino_esp32_core_version"
	git apply -v "$project_dir_configs/arduino/packages/esp32/$arduino_esp32_core_version/python.patch"
	assert_error $? "Failed to configure \"$arduino_esp32_core_name $arduino_esp32_core_version\""
}

function install_arduino_cmake_toolchain()
{
	# Verify installation
	if [ -d "$project_dir_cmake/$arduino_cmake_toolchain_name" ]
	then
		echo "[INFO]: \"$arduino_cmake_toolchain_name\" installation exists. Skip."
		return
	fi

	# Create temporary directory
	local temp_dir
	temp_dir="$(mktemp -d /tmp/$project_name-XXXXXXXXXX)"
	assert_error $? "Failed to create temporary directory"

	# Download Arduino CMake Toolchain
	echo "[INFO]: Downloading \"$arduino_cmake_toolchain_name $arduino_cmake_toolchain_version_short\"..."
	cd "$temp_dir"
	git clone "$arduino_cmake_toolchain_url" "$arduino_cmake_toolchain_name" --branch "release-1.1-dev"
	assert_warn $? "Failed to download \"$arduino_cmake_toolchain_name $arduino_cmake_toolchain_version_short\""

	# Configure Arduino CMake Toolchain
	echo "[INFO]: Configuring \"$arduino_cmake_toolchain_name $arduino_cmake_toolchain_version_short\"..."
	cd "$temp_dir/$arduino_cmake_toolchain_name"
	git checkout -q "$arduino_cmake_toolchain_version"
	git apply -v "$project_dir_configs/$arduino_cmake_toolchain_name/arduino_cli_detection.patch"
	git apply -v "$project_dir_configs/$arduino_cmake_toolchain_name/compiler_sdk_path.patch"
	git apply -v "$project_dir_configs/$arduino_cmake_toolchain_name/property_resolution.patch"
	git apply -v "$project_dir_configs/$arduino_cmake_toolchain_name/general.patch"
	assert_warn $? "Failed to configure \"$arduino_cmake_toolchain_name $arduino_cmake_toolchain_version_short\""

	# Install Arduino CMake Toolchain
	echo "[INFO]: Installing \"$arduino_cmake_toolchain_name $arduino_cmake_toolchain_version_short\"..."
	mv "$temp_dir/$arduino_cmake_toolchain_name" "$project_dir_cmake/$arduino_cmake_toolchain_name"
	assert_warn $? "Failed to install \"$arduino_cmake_toolchain_name $arduino_cmake_toolchain_version_short\""
	echo "[INFO]: Finished installing \"$arduino_cmake_toolchain_name $arduino_cmake_toolchain_version_short\"."

	# Remove temporary directory
	rm -rf "$temp_dir"
	assert_warn $? "Failed to remove temporary directory"
}

function install_toolchain()
{
	echo "[INFO]: Installing \"$project_name\" toolchain..."

	# Install Arduino CLI
	install_arduino_cli

	# Install Arduino ESP32 Core
	install_arduino_esp32_core

	# Install Arduino CMake Toolchain
	install_arduino_cmake_toolchain

	echo "[INFO]: Finished installing \"$project_name\" toolchain."
}

function install_mcp23018()
{
	# Verify installation
	if [ -d "$project_dir_libraries/$mcp23018_name" ]
	then
		echo "[INFO]: \"$mcp23018_name\" installation exists. Skip."
		return
	fi

	# Create temporary directory
	local temp_dir
	temp_dir="$(mktemp -d /tmp/$project_name-XXXXXXXXXX)"
	assert_error $? "Failed to create temporary directory"

	# Download MCP23018
	echo "[INFO]: Downloading \"$mcp23018_name $mcp23018_version_short\"..."
	cd "$temp_dir"
	git clone "$mcp23018_url" "$mcp23018_name" --branch "master"
	assert_warn $? "Failed to download \"$mcp23018_name $mcp23018_version_short\""

	# Configure MCP23018
	echo "[INFO]: Configuring \"$mcp23018_name $mcp23018_version_short\"..."
	cd "$temp_dir/$mcp23018_name/libraries/$mcp23018_name"
	git checkout -q "$mcp23018_version"
	git apply -v "$project_dir_configs/libraries/$mcp23018_name/esp32.patch"
	git apply -v "$project_dir_configs/libraries/$mcp23018_name/i2c.patch"
	assert_warn $? "Failed to configure \"$mcp23018_name $mcp23018_version_short\""

	# Install MCP23018
	echo "[INFO]: Installing \"$mcp23018_name $mcp23018_version_short\"..."
	mkdir -p "$project_dir_libraries/$mcp23018_name"
	mv "$temp_dir/$mcp23018_name/libraries/$mcp23018_name/$mcp23018_name"* "$project_dir_libraries/$mcp23018_name"
	touch "$project_dir_libraries/$mcp23018_name/library.properties"
	echo "name=$mcp23018_name" > "$project_dir_libraries/$mcp23018_name/library.properties"
	assert_warn $? "Failed to install \"$mcp23018_name $mcp23018_version_short\""
	echo "[INFO]: Finished installing \"$mcp23018_name $mcp23018_version_short\"."

	# Remove temporary directory
	rm -rf "$temp_dir"
	assert_warn $? "Failed to remove temporary directory"
}

function configure_kalman_filter_library()
{
	# Verify installation
	if [ ! -d "$project_dir_libraries/$kalman_filter_library_name" ]
	then
		echo "[ERROR]: \"$kalman_filter_library_name\" installation not found. Quit."
		exit 1
	fi

	# Configure Kalman Filter Library
	echo "[INFO]: Configuring \"$kalman_filter_library_name $kalman_filter_library_version\"..."
	cd "$project_dir_libraries/$kalman_filter_library_name"
	git apply -v "$project_dir_configs/libraries/$kalman_filter_library_name/double.patch"
	assert_error $? "Failed to configure \"$kalman_filter_library_name $kalman_filter_library_version\""
}

function configure_stm32duino_vl53l4cx()
{
	# Verify installation
	if [ ! -d "$project_dir_libraries/$stm32duino_vl53l4cx_name" ]
	then
		echo "[ERROR]: \"$stm32duino_vl53l4cx_name\" installation not found. Quit."
		exit 1
	fi

	# Configure STM32duino VL53L4CX
	echo "[INFO]: Configuring \"$stm32duino_vl53l4cx_name $stm32duino_vl53l4cx_version\"..."
	cd "$project_dir_libraries/$stm32duino_vl53l4cx_name/src"
	git apply -v "$project_dir_configs/libraries/$stm32duino_vl53l4cx_name/io_expander.patch"
	assert_error $? "Failed to configure \"$stm32duino_vl53l4cx_name $stm32duino_vl53l4cx_version\""
}

function install_libraries()
{
	echo "[INFO]: Installing \"$project_name\" libraries..."

	# Install Adafruit BusIO
	install_arduino_library "$adafruit_busio_name" "$adafruit_busio_version" "$adafruit_busio_author"

	# Install Adafruit GFX Library
	install_arduino_library "$adafruit_gfx_library_name" "$adafruit_gfx_library_version" "$adafruit_gfx_library_author"

	# Install Adafruit MPU6050
	install_arduino_library "$adafruit_mpu6050_name" "$adafruit_mpu6050_version" "$adafruit_mpu6050_author"

	# Install Adafruit NeoPixel
	install_arduino_library "$adafruit_neopixel_name" "$adafruit_neopixel_version" "$adafruit_neopixel_author"

	# Install Adafruit SH110X
	install_arduino_library "$adafruit_sh110x_name" "$adafruit_sh110x_version" "$adafruit_sh110x_author"

	# Install Adafruit Unified Sensor
	install_arduino_library "$adafruit_unified_sensor_name" "$adafruit_unified_sensor_version" "$adafruit_unified_sensor_author"

	# Install Eigen
	install_arduino_library "$eigen_name" "$eigen_version" "$eigen_author"

	# Install Kalman Filter Library
	install_arduino_library "$kalman_filter_library_name" "$kalman_filter_library_version" "$kalman_filter_library_author" "configure_kalman_filter_library"

	# Install MCP23018
	install_mcp23018

	# Install STM32duino VL53L4CX
	install_arduino_library "$stm32duino_vl53l4cx_name" "$stm32duino_vl53l4cx_version" "$stm32duino_vl53l4cx_author" "configure_stm32duino_vl53l4cx"

	echo "[INFO]: Finished installing \"$project_name\" libraries."
}

function create_project()
{
	# Verify project
	if [ -d "$project_dir_build/$project_name" ]
	then
		echo "[INFO]: \"$project_name\" project exists. Skip."
		return
	fi

	# Create project directory
	mkdir -p "$project_dir_build/$project_name"
	assert_error $? "Failed to create \"$project_name\" project directory"

	# Create project
	echo "[INFO]: Creating \"$project_name\" project..."
	cd "$project_dir_build/$project_name"
	cmake -G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_BUILD_TYPE="Release" -DCMAKE_ECLIPSE_MAKE_ARGUMENTS="-j$cpu_cores" -DCMAKE_ECLIPSE_VERSION="$eclipse_version" -DCMAKE_TOOLCHAIN_FILE="$project_dir_cmake/$arduino_cmake_toolchain_name/Arduino-toolchain.cmake" -DARDUINO_BOARD_OPTIONS_FILE="$project_dir_cmake/$project_name/BoardOptions.cmake" "$project_dir_src/$project_name"
	assert_error $? "Failed to create \"$project_name\" project"

	echo "[INFO]: Finished creating \"$project_name\" project."
}

function main()
{
	# Validate operating system
	validate_os

	# Install toolchain
	install_toolchain

	# Install libraries
	install_libraries

	# Create project
	create_project

	echo "[INFO]: Finished seting up \"$project_name\"."
}

# Call main function
main "$@"
