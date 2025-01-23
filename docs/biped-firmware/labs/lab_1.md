## Lab 1: Setup and I/O

In this lab, you will set up the Biped firmware project development environment and implement basic I/O operations in the Biped firmware.

Before getting started with this lab, first go through the entire project [README](../../../../../README.md), and then go through the entire [Eclipse Guide](../../biped-firmware/general/eclipse.md). Additionally, review the [Biped Guide](../../biped-firmware/general/biped.md) on how to properly handle and operate Biped.

### Objectives

1. Implement all Lab 1 todos in `main.cpp`.
2. Implement all Lab 1 todos in `task.cpp`.

Read carefully the comment blocks provided in the above header and source files for detailed steps and instructions. Search for `TODO LAB 1 YOUR CODE HERE` in the above header and source files to locate the todos.

### Demo (100 points)

1. (50 points) Demonstrate the correctness of all OLED display I/O operations. The Biped serial number, real-time task timings, heap utilization percentage, Wi-Fi status, controller status, and planner status should be printed to the OLED display on the correct lines.
2. (25 points) Demonstrate the correctness of all serial I/O operations. Upon boot, the Biped firmware initialization status should be printed to the serial terminal.
3. (25 points) Demonstrate the correctness of all NeoPixel I/O operations. The NeoPixel array should light up with the default color (blue).

If a lab group fails to complete the demo, partial credits will be given at the TA's discretion by examining the completeness and correctness of the lab group's codebase submitted to Canvas.

Written by Simon Yu.
