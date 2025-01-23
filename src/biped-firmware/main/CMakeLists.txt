# Add executables
add_executable(
	biped-firmware
	main.cpp
)

# Set library link targets
target_link_libraries(
	biped-firmware PRIVATE
	biped-firmware-common
	biped-firmware-task
)

# Set arduino library link targets
target_link_arduino_libraries(
	biped-firmware PRIVATE
	core
	EEPROM
	SPI
	Wire
)

# Set arduino upload targets
target_enable_arduino_upload(biped-firmware)
