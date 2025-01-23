## Lab 3: FreeRTOS

In this lab, you will set up the FreeRTOS task framework in the Biped firmware.

### Objectives

1. Implement all Lab 3 todos in `interrupt.cpp`.
2. Implement all Lab 3 todos in `main.cpp`.
3. Implement all Lab 3 todos in `task.cpp`.

Read carefully the comment blocks provided in the above header and source files for detailed steps and instructions. Search for `TODO LAB 3 YOUR CODE HERE` in the above header and source files to locate the todos.

### Demo (100 points)

Demonstrate the correctness of the FreeRTOS task framework. At this point, the real-time task remains largely unimplemented. Therefore, add delays in the real-time task using the Arduino function `delayMicroseconds` to simulate real-time task workloads. Ensure the delay is well below the real-time task interval, i.e., the fast domain period. The left entry of the real-time task timings printed to the OLED display, i.e., the real-time task execution time, should now display the execution time of the real-time task, i.e., the delays added in the real-time task, plus some fluctuating overheads. Remember to remove all added delays after the demo.

Note: the correctness of this lab can only be fully verified in subsequent labs. Earning full credits for this lab does not necessarily indicate the complete functional correctness of your FreeRTOS implementations. It is advised to double-check your implementations for this lab and consult with the TA, if necessary, to ensure your success in subsequent labs.

If a lab group fails to complete the demo, partial credits will be given at the TA's discretion by examining the completeness and correctness of the lab group's codebase submitted to Canvas.

Written by Simon Yu.
