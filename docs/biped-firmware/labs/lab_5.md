## Lab 5: Biped Ground Station

In this lab, you will implement the FreeRTOS tasks responsible for the communications between the Biped ground station and the Biped firmware.

Follow the Biped ground station project README to set up and build the Biped ground station. The Biped ground station user interface should be intuitive and self-explanatory; however, please get in touch with the TA with any questions or doubts about using the Biped ground station.

### Objectives

1. Implement all Lab 5 todos in `main.cpp`.
2. Implement all Lab 5 todos in `task.cpp`.

Read carefully the comment blocks provided in the above header and source files for detailed steps and instructions. Search for `TODO LAB 5 YOUR CODE HERE` in the above header and source files to locate the todos.

### Demo (100 points)

On the lab workstation, run the `ip addr` command, locate the network interface `eno0`, and note down its IP address as the workstation IP address on the CS 431 wireless local network. Set this IP address to the `ip_ground_station` network parameter in the Biped firmware parameter header. Then, re-comple and re-flash the Biped firmware.

Note: Biped may fail to establish the Wi-Fi connection after flashing. In this case, press the reset button to trigger a soft reset. If the Wi-Fi connection issue persists, double-check the implementation of the network task.

1. (25 points) Demonstrate the correctness of the network task. The Wi-Fi connection should be correctly initialized and established, with the Wi-Fi local IP address printed on the OLED display.

In the Biped ground station, select the `Settings` tab and set the Biped IP address as the Wi-Fi local IP address printed on the Biped OLED display.

2. (25 points) Demonstrate the correctness of the Biped message UDP read task. Using the `Controller` tab in the Biped ground station, choose a random controller gain, update it with a random value, and then send (apply) the updated controller gain to Biped. In the `udpReadBipedMessageTask` function, print the chosen controller gain to the serial. Do not print to the OLED display. The chosen controller gain printed by Biped should match the value sent from the Biped ground station. Remember to remove all added prints after the demo.
3. (25 points) Demonstrate the correctness of the Biped message UDP write task. Using the `Data` tab in the Biped ground station, the `Sequence` and `Timestamp` entries under `Header` should now be constantly incrementing.
4. (25 points) Demonstrate the correctness of the camera UDP write task. The continuously streamed camera images should now be viewable under the `Camera` tab in the Biped ground station.

Note: make sure only one Biped ground station instance is running at any given time. A running Biped ground station will lock the UDP port, preventing any other instances from accessing the same port.

Note: incorrect implementation of the FreeRTOS task framework in Lab 3 may affect the communication with the Biped ground station. Double-check your FreeRTOS task framework implementations.

Note: if you were unable to get the Biped ground station to function or work reliably, credits will be given at the TA's discretion by examining the completeness and correctness of the lab group's codebase submitted to Canvas.

If a lab group fails to complete the demo, partial credits will be given at the TA's discretion by examining the completeness and correctness of the lab group's codebase submitted to Canvas.

Written by Simon Yu.
