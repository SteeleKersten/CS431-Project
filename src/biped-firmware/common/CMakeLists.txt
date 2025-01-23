# Add libraries
add_library(
	biped-firmware-common
	global.cpp
)

# Set library link targets
target_link_libraries(
	biped-firmware-common PRIVATE
	biped-firmware-actuator
	biped-firmware-controller
	biped-firmware-network
	biped-firmware-planner
	biped-firmware-platform
	biped-firmware-sensor
	biped-firmware-utility
)
