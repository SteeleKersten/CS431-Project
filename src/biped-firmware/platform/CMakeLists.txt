# Add libraries
add_library(
	biped-firmware-platform
	camera.cpp
	display.cpp
	encoder.cpp
	imu.cpp
	io_expander.cpp
	neopixel.cpp
	serial.cpp
	time_of_flight.cpp
	timer.cpp
	wifi.cpp
)

# Set library link targets
target_link_libraries(
	biped-firmware-platform PRIVATE
	biped-firmware-common
	biped-firmware-sensor
	biped-firmware-utility
)

# Set arduino library link targets
target_link_arduino_libraries(
	biped-firmware-platform PRIVATE
	core
	Wire
	"Adafruit BusIO"
	"Adafruit MPU6050"
	"Adafruit GFX Library"
	"Adafruit NeoPixel"
	"Adafruit SH110X"
	"Adafruit Unified Sensor"
	"MCP23018"
	"Kalman Filter Library"
	"STM32duino VL53L4CX"
)
