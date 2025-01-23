# Set global parameters
cmake_minimum_required(VERSION 3.7.0)
project(biped-firmware)

# Set compiler parameters
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add library extra search paths
list(APPEND ARDUINO_LIBRARIES_SEARCH_PATHS_EXTRA "${CMAKE_SOURCE_DIR}/..")

# Disable compiler built-in include directory filtering
unset(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES)
unset(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)

# Set installation paths
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../../install/${CMAKE_PROJECT_NAME}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../../install/${CMAKE_PROJECT_NAME}/lib")

# Set include directories
include_directories(
	"${CMAKE_SOURCE_DIR}"
	"${CMAKE_BINARY_DIR}"
	"${ARDUINO_BOARD_RUNTIME_PLATFORM_PATH}/cores/esp32"
	"${ARDUINO_BOARD_RUNTIME_PLATFORM_PATH}/libraries/EEPROM/src"
	"${ARDUINO_BOARD_RUNTIME_PLATFORM_PATH}/libraries/SPI/src"
	"${ARDUINO_BOARD_RUNTIME_PLATFORM_PATH}/libraries/WiFi/src"
	"${ARDUINO_BOARD_RUNTIME_PLATFORM_PATH}/libraries/Wire/src"
	"${ARDUINO_BOARD_RUNTIME_PLATFORM_PATH}/variants/esp32"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/config"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/driver/include"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/esp_common/include"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/esp_http_server/include"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/esp_hw_support/include"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/esp_timer/include"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/esp32-camera/conversions/include"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/esp32-camera/driver/include"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/fb_gfx/include"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/freertos/include"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/freertos/include/esp_additions"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/freertos/include/esp_additions/freertos"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/freertos/port/xtensa/include"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/hal/include"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/hal/platform_port/include"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/log/include"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/nghttp/port/include"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/soc/esp32/include"
	"${ARDUINO_BOARD_COMPILER_SDK_PATH}/include/soc/esp32/include/soc"
	"${CMAKE_SOURCE_DIR}/../libraries/Adafruit_BusIO"
	"${CMAKE_SOURCE_DIR}/../libraries/Adafruit_GFX_Library"
	"${CMAKE_SOURCE_DIR}/../libraries/Adafruit_MPU6050"
	"${CMAKE_SOURCE_DIR}/../libraries/Adafruit_NeoPixel"
	"${CMAKE_SOURCE_DIR}/../libraries/Adafruit_SH110X"
	"${CMAKE_SOURCE_DIR}/../libraries/Adafruit_Unified_Sensor"
	"${CMAKE_SOURCE_DIR}/../libraries/Eigen"
	"${CMAKE_SOURCE_DIR}/../libraries/Kalman_Filter_Library"
	"${CMAKE_SOURCE_DIR}/../libraries/MCP23018"
	"${CMAKE_SOURCE_DIR}/../libraries/STM32duino_VL53L4CX/src"
)

# Add subdirectories
add_subdirectory(actuator)
add_subdirectory(common)
add_subdirectory(controller)
add_subdirectory(main)
add_subdirectory(network)
add_subdirectory(planner)
add_subdirectory(platform)
add_subdirectory(sensor)
add_subdirectory(task)
add_subdirectory(utility)
