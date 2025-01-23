# Add libraries
add_library(
	biped-firmware-task
	interrupt.cpp
	task.cpp
)

# Set library link targets
target_link_libraries(
	biped-firmware-task PRIVATE
	biped-firmware-common
	biped-firmware-network
	biped-firmware-platform
)

# Set arduino library link targets
target_link_arduino_libraries(
	biped-firmware-task PRIVATE
	core
)
