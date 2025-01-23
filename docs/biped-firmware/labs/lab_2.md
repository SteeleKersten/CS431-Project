## Lab 2: Hardware Timer

In this lab, you will implement the hardware timer and timer interrupt in the Biped firmware.

Note: students are forbidden to use any ESP-IDF or any Arduino hardware timer functions or hardware timer API calls for this lab. The hardware timer implementation must use only raw registers per the instructions provided by the comment blocks for this lab.

### Objectives

1. Implement all Lab 2 todos in `interrupt.cpp`.
2. Implement all Lab 2 todos in `main.cpp`.
3. Implement all Lab 2 todos in `timer.cpp`.

Read carefully the comment blocks provided in the above header and source files for detailed steps and instructions. Search for `TODO LAB 2 YOUR CODE HERE` in the above header and source files to locate the todos.

### Demo (100 points)

Demonstrate the correctness of the hardware timer and timer interrupt. The right entry of the real-time task timings printed to the OLED display, i.e., the real-time task interval, should now display the fast domain period in microseconds. The displayed real-time task interval should be stable with minimal fluctuations around the set timer interval, i.e., the fast domain period.

If a lab group fails to complete the demo, partial credits will be given at the TA's discretion by examining the completeness and correctness of the lab group's codebase submitted to Canvas.

Written by Simon Yu.
