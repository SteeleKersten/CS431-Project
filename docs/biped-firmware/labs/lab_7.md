## Lab 7: Control

In this lab, you will implement a set of controllers in the Biped firmware.

Note: the controller gains are specific to each individual Bipeds due to the subtle differences in their hardware construction, e.g., the center of gravity. Therefore, from this point on, pick a Biped and stick to the same one for the remaining labs. Notify the TA immediately regarding any loose components, as they may affect the control performance.

Note: if you were unable to get the Biped ground station to function or work reliably, with the TA's approval, perform the controller tuning by directly changing the gain values in the firmware source code, and re-compile and re-flash the firmware every time you modify the gain values.

### Objectives

1. Implement all Lab 7 todos in `controller.cpp`.
2. Implement all Lab 7 todos in `main.cpp`.
3. Implement all Lab 7 todos in `open_loop_controller.cpp`.
4. Implement all Lab 7 todos in `pid_controller.cpp`.
5. Implement all Lab 7 todos in `task.cpp`.

Read carefully the comment blocks provided in the above header and source files for detailed steps and instructions. Search for `TODO LAB 7 YOUR CODE HERE` in the above header and source files to locate the todos.

### Demo (100 points)

Using the `Controller` tab in the Biped ground station, demonstrate the controller performance using the controller response plots.

1. (50 points) Demonstrate the correctness of the pitch controller by having Biped attempt to balance itself. It is normal if Biped tries to run off along the X axis, as the pitch controller alone cannot hold the Biped in place.
2. (50 points) Demonstrate the correctness of the position controller by showing Biped returning to the initial position when pushed away along the X axis.

It is recommended to perform the above two demos at once, i.e., by running both controllers together instead of separately. The controllers are considered well-tuned if Biped is able to compensate light to medium push to the blue aluminum base chassis plate along the X axis.

Note: if you were unable to get the Biped ground station to function or work reliably, continue with the demos without the Biped ground station.

Note: make sure only one Biped ground station instance is running at any given time. A running Biped ground station will lock the UDP port, preventing any other instances from accessing the same port.

If a lab group fails to complete the demo, partial credits will be given at the TA's discretion by examining the completeness and correctness of the lab group's codebase submitted to Canvas.

Written by Simon Yu.
