# Add libraries
add_library(
	biped-firmware-actuator
	actuator.cpp
)

# Set library link targets
target_link_libraries(
	biped-firmware-actuator PRIVATE
	biped-firmware-common
	biped-firmware-controller
	biped-firmware-utility
)

# Set arduino library link targets
target_link_arduino_libraries(
	biped-firmware-actuator PRIVATE
	core
)
