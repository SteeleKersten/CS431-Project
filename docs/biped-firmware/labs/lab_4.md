## Lab 4: I/O Expander

In this lab, you will set up the I/O expander digital I/O and interrupt service framework in the Biped firmware.

### Objectives

1. Implement all Lab 4 todos in `interrupt.cpp`.
2. Implement all Lab 4 todos in `io_expander.cpp`.
3. Implement all Lab 4 todos in `main.cpp`.
4. Implement all Lab 4 todos in `task.cpp`.

Read carefully the comment blocks provided in the above header and source files for detailed steps and instructions. Search for `TODO LAB 4 YOUR CODE HERE` in the above header and source files to locate the todos.

### Demo (100 points)

1. (25 points) Demonstrate the correctness of the I/O expander digital I/O framework by polling the push button A pin and printing the pin state to either the serial or the OLED display in the `bestEffortTask` function. Remember to remove all added prints after the demo.
2. (25 points) Demonstrate the correctness of edge-triggered interrupt handling of the I/O expander interrupt service framework by incrementing the Biped serial number global variable in the push button A interrupt handler. The Biped serial number should be incremented with every push button A press (as soon as the button is pressed down). The real-time task should continue to function (the real-time task timings should continue to update on the OLED display) when the button is held down. Remember to remove the Biped serial number increments after the demo.
3. (25 points) Demonstrate the correctness of level-triggered interrupt handling of the I/O expander interrupt service framework by incrementing the Biped serial number global variable in the push button A interrupt handler. This time, also modify the interrupt mode of the push button A interrupt handler to be on-low mode. The Biped serial number should be incremented as soon as the button is released. The remaining FreeRTOS tasks, however, should cease to function (e.g., the OLED display should cease to update) when the button is held down. Why is this the case? Remember to remove the Biped serial number increments and restore the interrupt mode of the push button A interrupt handler after the demo.
4. (25 points) Demonstrate the correctness of the I/O expander digital I/O and interrupt service framework for I/O expander B (the expansion header pins). Choose a random pin on the I/O expander B and repeat the first three demos. Set up the pin mode and attach an interrupt handler for the chosen pin. The TA will connect an external push button to the chosen pin on the expansion header for this demo. Remember to restore the codebase (remove all added statements associated with the randomly chosen I/O expander B pin) after the demo.

Note: incorrect implementation of the FreeRTOS task framework in Lab 3 may cause issues with the I/O expander interrupt service framework implementation. Double-check your FreeRTOS task framework implementations.

If a lab group fails to complete the demo, partial credits will be given at the TA's discretion by examining the completeness and correctness of the lab group's codebase submitted to Canvas.

Written by Simon Yu.
