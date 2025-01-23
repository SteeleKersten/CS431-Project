/**
 *  @file   parameter.h
 *  @author Simon Yu
 *  @date   01/12/2022
 *  @brief  Parameter constant expression header.
 *
 *  This file defines the parameter constant expressions.
 */

/*
 *  Include guard.
 */
#ifndef COMMON_PARAMETER_H_
#define COMMON_PARAMETER_H_

/*
 *  External headers.
 */
#include <cstddef>
#include <cstdint>
#include <freertos/FreeRTOS.h>
#include <hal/timer_types.h>

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
/*
 *  Address parameter namespace.
 */
namespace AddressParameter
{
constexpr uint8_t display = 0x3C;   //!< OLED display I2C address.
constexpr uint8_t eeprom_serial_number = 0x00;  //!< Biped serial number EEPROM address.
constexpr uint8_t imu_mpu6050 = 0x69;   //!< MPU6050 IMU I2C address.
constexpr uint8_t io_expander_a = 0x00; //!< I/O expander A I2C address.
constexpr uint8_t io_expander_b = 0x07; //!< I/O expander B I2C address.
constexpr uint8_t time_of_flight_left = 0x40;   //!< Left time-of-flight I2C address.
constexpr uint8_t time_of_flight_middle = 0x41; //!< Middle time-of-flight I2C address.
constexpr uint8_t time_of_flight_right = 0x42;  //!< Right time-of-flight I2C address.
}   // namespace AddressParameter

/*
 *  Camera parameter namespace.
 */
namespace CameraParameter
{
constexpr size_t frame_buffer_count = 1;    //!< Number of frame buffers.
constexpr int jpeg_quality = 12;    //!< JPEG image quality.
constexpr int jpeg_quality_conversion = 80;    //!< JPEG image conversion quality.
constexpr int xclk_frequency = 20000000;    //!< X clock frequency, in Hertz.
}   // namespace CameraParameter

/*
 *  Display parameter namespace.
 */
namespace DisplayParameter
{
constexpr double acceleration_z_rotation = 0; //!< OLED display rotation X acceleration threshold, in meters per second squared.
constexpr uint16_t height = 64; //!< OLED display height, in pixels.
constexpr double low_pass_filter_beta = 0.9;    //!< Low-pass filter beta.
constexpr uint32_t postclk = 1000000;   //!< Post-clock frequency, in Hertz.
constexpr uint32_t preclk = 1000000;    //!< Pre-clock frequency, in Hertz.
constexpr int8_t reset_pin = -1;    //!< Reset pin.
constexpr uint8_t rotation_upright = 1; //!< Upright rotation.
constexpr uint8_t rotation_inverted = 3;    //!< Inverted rotation.
constexpr uint8_t text_size = 1;    //!< OLED display text size.
constexpr uint16_t width = 128; //!< OLED display width, in pixels.
}   // namespace DisplayParameter

/*
 *  EEPROM parameter namespace.
 */
namespace EEPROMParameter
{
constexpr size_t size = 4;  //!< EEPROM size, in bytes.
}   // namespace EEPROMParameter

/*
 *  Encoder parameter namespace.
 */
namespace EncoderParameter
{
constexpr double low_pass_filter_beta = 0.7;    //!< Low-pass filter beta.
constexpr unsigned steps_per_meter = 7400;  //!< Number of encoder steps per meter.
}   // namespace EncoderParameter

/*
 *  I/O expander parameter namespace.
 */
namespace IOExpanderParameter
{
constexpr size_t port_count = 2; //!< Number of ports.
constexpr size_t port_pin_count = 8; //!< Number of pins per port.
}   // namespace IOExpanderParameter

/*
 *  Kalman filter parameter namespace.
 */
namespace KalmanFilterParameter
{
constexpr float q_angle = 0.001;    //!< Q angle.
constexpr float q_bias = 0.005; //!< Q bias.
constexpr float r_measure = 0.5;    //!< R measure.
}   // namespace KalmanFilterParameter

/*
 *  Median filter parameter namespace.
 */
namespace MedianFilterParameter
{
constexpr size_t window_size = 10;  //!< Window size.
}   // namespace MedianFilterParameter

/*
 *  Motor parameter namespace.
 */
namespace MotorParameter
{
constexpr unsigned pwm_min = 0; //!< Minimum pulse-width modulation (PWM) value.
constexpr unsigned pwm_max = 255;   //!< Maximum pulse-width modulation (PWM) value.
}   // namespace MotorParameter

/*
 *  NeoPixel parameter namespace.
 */
namespace NeoPixelParameter
{
constexpr uint8_t brightness_max = 30;  //!< Maximum brightness.
constexpr uint8_t brightness_min = 10;  //!< Minimum brightness.
constexpr uint16_t size = 4;    //!< Size of the NeoPixel LED array.
}   // namespace NeoPixelParameter

/*
 *  Network parameter namespace.
 */
namespace NetworkParameter
{
constexpr size_t buffer_size_biped_message = 1024; //!< Biped message buffer size.
constexpr size_t buffer_size_camera = 1460; //!< Camera buffer size.
constexpr char camera_frame_boundary[] = "123456789000000000000987654321"; //!< Camera frame magic boundary string.
constexpr char ip_ground_station[] = "192.168.0.101";   //!< Biped ground station IP address.
constexpr char passphrase[] = "s4srxcF3";   //!< Wi-Fi passphrase.
constexpr uint16_t port_udp_biped_message = 4431;  //!< Biped message UDP port.
constexpr uint16_t port_udp_camera = 4432;  //!< Camera UDP port.
constexpr char ssid[] = "CS 431";    //!< Wi-Fi SSID.
}   // namespace NetworkParameter

/*
 *  Period parameter namespace.
 */
namespace PeriodParameter
{
/*
 *  The fast domain period is used by the hardware timer for
 *  generating timer interrupts for the real-time task.
 *
 *  The slow domain period specifies how often the slow domain
 *  tasks run within the real-time task function. The slow
 *  domain period is always a multiple of fast domain period.
 *
 *  Refer to the real-time task function in the task header
 *  for more details.
 */
constexpr double fast = 0.005;  //!< Fast domain period, in seconds.
constexpr double slow = 0.04;   //!< Slow domain period, in seconds.
}   // namespace PeriodParameter

/*
 *  Serial parameter namespace.
 */
namespace SerialParameter
{
constexpr unsigned long baud_rate = 115200; //!< Baud rate.
constexpr LogLevel log_level_max = LogLevel::trace; //!< Maximum log level.
}   // namespace SerialParameter

/*
 *  Task parameter namespace.
 */
namespace TaskParameter
{
constexpr int core_0 = 0;   //!< CPU core 0.
constexpr int core_1 = 1;   //!< CPU core 1.
constexpr unsigned int priority_min = 0;    //!< Minimum priority.
constexpr unsigned int priority_max = configMAX_PRIORITIES; //!< Maximum priority.
constexpr uint32_t stack_size = 4096;   //!< Task stack size, in bytes.
}   // namespace TaskParameter

/*
 *  Timer parameter namespace.
 */
namespace TimerParameter
{
constexpr timer_group_t group = TIMER_GROUP_1; //!< Timer group.
constexpr timer_idx_t index = TIMER_1; //!< Timer index.
constexpr size_t prescaler = 80;  //!< Timer prescaler.
}   // namespace TimerParameter
}   // namespace firmware
}   // namespace biped

#endif  // COMMON_PARAMETER_H_
