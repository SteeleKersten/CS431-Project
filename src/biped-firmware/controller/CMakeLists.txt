# Add libraries
add_library(
	biped-firmware-controller
	controller.cpp
	open_loop_controller.cpp
	pid_controller.cpp
)

# Set library link targets
target_link_libraries(
	biped-firmware-controller PRIVATE
	biped-firmware-common
	biped-firmware-platform
	biped-firmware-sensor
	biped-firmware-utility
)
