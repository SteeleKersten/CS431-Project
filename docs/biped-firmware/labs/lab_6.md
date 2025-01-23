## Lab 6: Sensing and Actuation

In this lab, you will implement sensing and actuation in the Biped firmware, i.e., reading data from sensors and performing motor actuation.

### Objectives

1. Implement all Lab 6 todos in `actuator.cpp`.
2. Implement all Lab 6 todos in `encoder.cpp`.
3. Implement all Lab 6 todos in `imu.cpp`.
4. Implement all Lab 6 todos in `interrupt.cpp`.
5. Implement all Lab 6 todos in `main.cpp`.
6. Implement all Lab 6 todos in `sensor.cpp`.
7. Implement all Lab 6 todos in `task.cpp`.

Read carefully the comment blocks provided in the above header and source files for detailed steps and instructions. Search for `TODO LAB 6 YOUR CODE HERE` in the above header and source files to locate the todos.

### Demo (100 points)

Using the `Data` tab in the Biped ground station, demonstrate the correctness of your encoder, IMU, and time-of-flight sensor data.

1. (25 points) Demonstrate the correctness of all encoder data by pushing the Biped on a surface along its X axis. The magnitude and the signedness of all data should match the definition of the standard body reference frame.
2. (25 points) Demonstrate the correctness of all IMU data by carefully tiling the Biped by hand. The magnitude and the signedness of all data should match the definition of the standard body reference frame.
3. (25 points) Demonstrate the correctness of all time-of-flight data by placing objects in front of the Biped. The magnitude of the range data should reflect the distance from the time-of-flight sensors to the object, in meters.

Place the Biped upside-down. Yes, the OLED display now rotates by gravity, assuming the IMU data acquisition was implemented correctly. Contact the TA to obtain a motor jumper for the following demo and the subsequent labs. Place the motor jumper in the off position. Refer to the [Biped Guide](../general/biped.md#buttons-and-switches) for more details on how to operate the motor jumper.

4. (25 points) Demonstrate the correctness of the actuation of both motors by creating and initializing an `ActuationCommand` struct in the `bestEffortTask` function, and then actuating the motors by modifying the values in the `ActuationCommand` struct and setting the struct to the actuator. The Biped should move forward with PWM values above the minimum PWM value, and vice versa.

Note: if you were unable to get the Biped ground station to function or work reliably, with the TA's approval, demonstrate the correctness of your encoder, IMU, and time-of-flight sensor data by printing them to the serial or the OLED display in the `bestEffortTask` function. In this case, please print all angle values as degrees. When printing to serial, add a small delay after the serial prints for improved readability. Remember to remove all added prints and delays after the demo.

Note: make sure only one Biped ground station instance is running at any given time. A running Biped ground station will lock the UDP port, preventing any other instances from accessing the same port.

If a lab group fails to complete the demo, partial credits will be given at the TA's discretion by examining the completeness and correctness of the lab group's codebase submitted to Canvas.

Written by Simon Yu.
