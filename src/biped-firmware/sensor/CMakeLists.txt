# Add libraries
add_library(
	biped-firmware-sensor
	sensor.cpp
)

# Set library link targets
target_link_libraries(
	biped-firmware-sensor PRIVATE
	biped-firmware-common
	biped-firmware-platform
)
