# Add libraries
add_library(
	biped-firmware-planner
	maneuver_planner.cpp
	waypoint_planner.cpp
)

# Set library link targets
target_link_libraries(
	biped-firmware-planner PRIVATE
	biped-firmware-common
	biped-firmware-controller
	biped-firmware-platform
	biped-firmware-sensor
	biped-firmware-utility
)
