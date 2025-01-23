# Add libraries
add_library(
	biped-firmware-utility
	math.cpp
)

# Set arduino library link targets
target_link_arduino_libraries(
	biped-firmware-utility PRIVATE
	core
)
