## Lab 8: Planning

In this lab, you will implement a waypoint-based planner and a maneuver-based planner in the Biped firmware. Additionally, you will fine-tune your controllers using the planners.

Note: the controller gains are specific to each individual Bipeds due to the subtle differences in their hardware construction, e.g., the center of gravity. Therefore, from this point on, pick a Biped and stick to the same one for the remaining labs. Notify the TA immediately regarding any loose components, as they may affect the control performance.

### Objectives

1. Implement all Lab 8 todos in `controller.cpp`.
2. Implement all Lab 8 todos in `interrupt.cpp`.
3. Implement all Lab 8 todos in `main.cpp`.
4. Implement all Lab 8 todos in `maneuver_planner.cpp`.
5. Implement all Lab 8 todos in `task.cpp`.
6. Implement all Lab 8 todos in `waypoint_planner.cpp`.

Read carefully the comment blocks provided in the above header and source files for detailed steps and instructions. Search for `TODO LAB 8 YOUR CODE HERE` in the above header and source files to locate the todos.

### Demo (100 points)

1. (50 points) Demonstrate the correctness of the waypoint planner by executing the waypoint planner example plan. The Biped should move forward and in reverse with moderate speeds and turn towards any yaw controller references given by the plan.
2. (50 points) Demonstrate the correctness of the maneuver planner by executing the maneuver planner example plan. The Biped should move forward and in reverse with moderate speeds and turn towards any yaw controller references given by the plan.

If a lab group fails to complete the demo, partial credits will be given at the TA's discretion by examining the completeness and correctness of the lab group's codebase submitted to Canvas.

Written by Simon Yu.
