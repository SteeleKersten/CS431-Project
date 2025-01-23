# Add libraries
add_library(
	biped-firmware-network
	udp.cpp
)

# Set library link targets
target_link_libraries(
	biped-firmware-network PRIVATE
	biped-firmware-common
)

# Set arduino library link targets
target_link_arduino_libraries(
	biped-firmware-network PRIVATE
	core
	WiFi
)
