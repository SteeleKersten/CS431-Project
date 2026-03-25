/**
 *  @file   imu.cpp
 *  @author Simon Yu
 *  @date   01/05/2023
 *  @brief  IMU class source.
 *
 *  This file implements the IMU class.
 */

/*
 *  Project headers.
 */
#include "common/parameter.h"
#include "platform/imu.h"
#include "platform/serial.h"
#include "utility/math.h"

/*
 *  Biped namespace.
 */
namespace biped
{
/*
 *  Firmware namespace.
 */
namespace firmware
{
IMU::IMU()
{
    /*
     *  Initialize IMU.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
    initialize();
}

IMUData
IMU::getData() const
{
    /*
     *  Return the class member MPU6050 IMU data struct.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
    return mpu6050_data_.IMUData;
}

void
IMU::read()
{
    /*
     *  Declare Adafruit sensor event structs.
     */
    sensors_event_t acceleration;
    sensors_event_t angular_velocity;
    sensors_event_t temperature;

    /*
     *  Using the getEvent function in the Adafruit_MPU6050 header, read
     *  from the MPU6050 IMU. Print an error message to the serial if the
     *  getEvent function returned false.
     *
     *  Refer to the Adafruit_MPU6050 header for the Adafruit MPU6050 IMU
     *  driver functions.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
    mpu6050_.getEvent(&acceleration, &angular_velocity, &temperature);
    
    /*
     *  Using the populated MPU6050 sensor event structs, populate
     *  the corresponding entries in the class member MPU6050 IMU data
     *  data struct.
     *
     *  Note that the raw data read from the MPU6050 might not be in the
     *  standard body reference frame. Refer to the following materials
     *  to correctly convert the raw data into the standard body reference
     *  frame. Assign zeros to the compass entries in the class member
     *  MPU6050 IMU data struct since MPU6050 does not have a compass.
     *
     *  Learn about the standard body reference frame here:
     *  https://www.vectornav.com/resources/inertial-navigation-primer/math-fundamentals/math-refframes
     *
     *  Learn about the rotational right-hand rule here:
     *  https://en.wikipedia.org/wiki/Right-hand_rule#Rotations
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
    mpu6050_data_.acceleration_x = acceleration.acceleration.x;
    mpu6050_data_.acceleration_y = acceleration.acceleration.y;
    mpu6050_data_.acceleration_z = acceleration.acceleration.z;
    mpu6050_data_.angular_velocity_x = angular_velocity.gyro.x;
    mpu6050_data_.angular_velocity_y = angular_velocity.gyro.y;
    mpu6050_data_.angular_velocity_z = angular_velocity.gyro.z;
    mpu6050_data_.temperature = temperature.temperature;
    mpu6050_data_.compass_x = 0;
    mpu6050_data_.compass_y = 0;
    mpu6050_data_.compass_z = 0;

    /*
     *  Calculate the attitude.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
    calculateAttitude();
}

void
IMU::initialize()
{
    /*
     *  Initialize the Adafruit MPU6050 IMU driver object. Leave wire and sensor ID
     *  function arguments as defaults. Print an error message to the serial if the
     *  initialization function returned false.
     *
     *  Refer to the Adafruit_MPU6050 header for Adafruit MPU6050 IMU driver
     *  functions and the parameter header for the MPU6050 IMU address parameter.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
    if (!mpu6050_.begin(AddressParameter::imu_mpu6050))
    {
        Serial(LogLevel::error) << "MPU6050 IMU initialization failed.";
        return;
    }

    /*
     *  Configure the Adafruit MPU6050 IMU driver object.
     */
    mpu6050_.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu6050_.setGyroRange(MPU6050_RANGE_250_DEG);

    /*
     *  Perform initial IMU read.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
    read();

    /*
     *  Perform initial Y attitude calculation and set the calculated
     *  attitude to the Y attitude in the class member MPU6050 IMU data
     *  struct.
     *
     *  Do not call the class member calculateAttitude function here.
     *  Instead, repeat the steps noted in the calculateAttitude function
     *  and directly calculate the Y attitude here. The initial Y attitude
     *  calculation should not be filtered by the Kalman filter as the Kalman
     *  filter has not yet been set up.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
    const double attitude_y_raw = atan2(-mpu6050_data_.acceleration_x, sqrt(pow(mpu6050_data_.acceleration_y, 2) + pow(mpu6050_data_.acceleration_z, 2)));
    mpu6050_data_.attitude_y = attitude_y_raw;

    /*
     *  Configure the Y attitude (pitch) Kalman filter.
     */
    mpu6050_kalman_filter_attitude_y_.setAngle(radiansToDegrees(mpu6050_data_.attitude_y));
    mpu6050_kalman_filter_attitude_y_.setQangle(KalmanFilterParameter::q_angle);
    mpu6050_kalman_filter_attitude_y_.setQbias(KalmanFilterParameter::q_bias);
    mpu6050_kalman_filter_attitude_y_.setRmeasure(KalmanFilterParameter::r_measure);
}

void
IMU::calculateAttitude()
{
    /*
     *  Validate fast period.
     */
    if (PeriodParameter::fast <= 0)
    {
        Serial(LogLevel::error) << "Invalid fast period.";
        return;
    }

    /*
     *  Calculate the raw Y attitude (pitch) data using the linear accelerations
     *  in the class member MPU6050 IMU data struct. Refer to the linked materials
     *  to correctly convert the calculated data into the standard body reference
     *  frame.
     *
     *  Note that the attitudes (roll, pitch, and yaw) are angles between pairs of
     *  acceleration vectors. Use atan2 function instead of atan for the correct
     *  signedness.
     *
     *  Remember to perform the same calculation in the initialize function as the
     *  initial Y attitude calculation. However, the initial Y attitude calculation
     *  in the initialize function should not be filtered by the Kalman filter as
     *  the Kalman filter has not yet been set up at that point.
     *
     *  Learn about the standard body reference frame here:
     *  https://www.vectornav.com/resources/inertial-navigation-primer/math-fundamentals/math-refframes
     *
     *  Learn about the rotational right-hand rule here:
     *  https://en.wikipedia.org/wiki/Right-hand_rule#Rotations
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
    const double attitude_y_raw = atan2(-mpu6050_data_.acceleration_x, sqrt(pow(mpu6050_data_.acceleration_y, 2) + pow(mpu6050_data_.acceleration_z, 2)));


    /*
     *  Filter the raw Y attitude data using the Kalman filter.
     */
    const double attitude_y_kalman_filter = mpu6050_kalman_filter_attitude_y_.getAngle(
            radiansToDegrees(attitude_y_raw), radiansToDegrees(mpu6050_data_.angular_velocity_y),
            PeriodParameter::fast);

    /*
     *  Convert the filtered Y attitude data back to radians
     *  using the degreesToRadians function in the math header and populate
     *  the Y attitude in the class member MPU6050 IMU data data struct.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
    mpu6050_data_.attitude_y = degreesToRadians(attitude_y_kalman_filter);

}
}   // namespace firmware
}   // namespace biped
