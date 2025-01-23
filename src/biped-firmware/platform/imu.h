/**
 *  @file   imu.h
 *  @author Simon Yu
 *  @date   01/05/2023
 *  @brief  IMU class header.
 *
 *  This file defines the IMU class.
 */

/*
 *  Include guard.
 */
#ifndef PLATFORM_IMU_H_
#define PLATFORM_IMU_H_

/*
 *  External headers.
 */
#include <Adafruit_MPU6050.h>
#include <Kalman.h>

/*
 *  Project headers.
 */
#include "common/type.h"

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
/**
 *  @brief  IMU class.
 *
 *  This class provides functions for reading from the
 *  inertial measurement unit (IM). The class also
 *  provides functions for calculating the attitude.
 */
class IMU
{
public:

    /**
     *  @brief  IMU class constructor.
     *
     *  This constructor initializes all class member variables.
     *  Additionally, the constructor initializes and configures
     *  the IMU driver.
     */
    IMU();

    /**
     *  @return IMU data struct.
     *  @brief  Get the class member IMU data struct.
     *
     *  This function returns the class member IMU data struct.
     */
    IMUData
    getData() const;

    /**
     *  @brief  IMU read function.
     *
     *  This function reads data from the IMU, populates the
     *  corresponding entries in the member IMU data struct,
     *  and calculates the attitude.
     */
    void
    read();

private:

    /**
     *  @brief  Initialize IMU.
     *
     *  This function initializes and configures the IMU driver, performs
     *  an initial read from the IMU and an initial attitude calculation,
     *  and configures the Kalman filters.
     */
    void
    initialize();

    /**
     *  @brief  Attitude calculation function.
     *
     *  This function calculates attitude data from linear acceleration
     *  data and populates the corresponding entries in the member
     *  IMU data struct. The function also filters attitude data using
     *  the Kalman filter.
     */
    void
    calculateAttitude();

    Adafruit_MPU6050 mpu6050_;  //!< Adafruit MPU6050 IMU driver object.
    IMUData mpu6050_data_;    //!< MPU6050 IMU data struct.
    Kalman mpu6050_kalman_filter_attitude_y_;  //!< MPU6050 Y attitude (pitch) Kalman filter object.
};
}   // namespace firmware
}   // namespace biped

#endif  // PLATFORM_IMU_H_
