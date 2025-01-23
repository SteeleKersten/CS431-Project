/**
 *  @file   pin.h
 *  @author Simon Yu
 *  @date   04/01/2022
 *  @brief  Pin constant expression header.
 *
 *  This file defines the pin constant expressions.
 */

/*
 *  Include guard.
 */
#ifndef COMMON_PIN_H_
#define COMMON_PIN_H_

/*
 *  External headers.
 */
#include <cstdint>

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
 *  ESP32 pin namespace.
 */
namespace ESP32Pin
{
constexpr int camera_d0 = 19;   //!< Camera digital line 0.
constexpr int camera_d1 = 22;   //!< Camera digital line 1.
constexpr int camera_d2 = 23;   //!< Camera digital line 2.
constexpr int camera_d3 = 21;   //!< Camera digital line 3.
constexpr int camera_d4 = 18;   //!< Camera digital line 4.
constexpr int camera_d5 = 17;   //!< Camera digital line 5.
constexpr int camera_d6 = 16;   //!< Camera digital line 6.
constexpr int camera_d7 = 2;    //!< Camera digital line 7.
constexpr int camera_href = 15; //!< Camera horizontal reference.
constexpr int camera_pclk = 5;  //!< Camera horizontal pixel clock.
constexpr int camera_pwdn = -1; //!< Camera power down.
constexpr int camera_reset = -1;    //!< Camera reset.
constexpr int camera_sscb_scl = 26; //!< Camera SCL.
constexpr int camera_sscb_sda = 14; //!< Camera SDA.
constexpr int camera_vsync = 13;    //!< Camera vertical sync.
constexpr int camera_xclk = 4;  //!< Camera X clock.
constexpr uint8_t io_expander_a_interrupt = 25; //!< I/O expander A interrupt.
constexpr uint8_t io_expander_b_interrupt = 33; //!< I/O expander B interrupt.
constexpr uint8_t i2c_scl = 26; //!< I2C SCL.
constexpr uint8_t i2c_sda = 14; //!< I2C SDA.
constexpr uint8_t motor_left_encoder_a = 39;    //!< Left motor encoder A.
constexpr uint8_t motor_left_encoder_b = 36;    //!< Left motor encoder B.
constexpr uint8_t motor_right_encoder_a = 35;   //!< Right motor encoder A.
constexpr uint8_t motor_right_encoder_b = 34;   //!< Right motor encoder B.
constexpr uint8_t motor_left_pwm = 27;  //!< Left motor PWM value.
constexpr uint8_t motor_right_pwm = 12; //!< Right motor PWM value.
constexpr uint8_t neopixel = 32;    //!< NeoPixel LED array.
}   // namespace ESP32Pin

/*
 *  I/O expander A Port A pin namespace.
 */
namespace IOExpanderAPortAPin
{
constexpr uint8_t motor_left_direction = 4; //!< Left motor direction.
constexpr uint8_t motor_right_direction = 3;  //!< Right motor direction.
constexpr uint8_t mpu6050_interrupt = 5;    //!< MPU6050 IMU interrupt.
constexpr uint8_t push_button_a = 6; //!< Push button A.
constexpr uint8_t push_button_b = 7; //!< Push button B.
constexpr uint8_t time_of_flight_left_interrupt = 1;    //!< Left time-of-flight interrupt.
constexpr uint8_t time_of_flight_left_shutdown = 0; //!< Left time-of-flight shutdown.
}   // namespace IOExpanderAPortAPin

/*
 *  I/O expander A Port B pin namespace.
 */
namespace IOExpanderAPortBPin
{
constexpr uint8_t bmx160_compass_interrupt_a = 3;   //!< BMX160 IMU compass interrupt A.
constexpr uint8_t bmx160_compass_interrupt_b = 4;   //!< BMX160 IMU compass interrupt B.
constexpr uint8_t motor_enable = 5;    //!< Motor enable.
constexpr uint8_t push_button_c = 0;    //!< Push button C.
constexpr uint8_t time_of_flight_middle_interrupt = 6;  //!< Middle time-of-flight interrupt.
constexpr uint8_t time_of_flight_middle_shutdown = 7;   //!< Middle time-of-flight shutdown.
constexpr uint8_t time_of_flight_right_interrupt = 2;   //!< Right time-of-flight interrupt.
constexpr uint8_t time_of_flight_right_shutdown = 1;    //!< Right time-of-flight shutdown.
}   // namespace IOExpanderAPortBPin

/*
 *  I/O expander B Port A pin namespace.
 */
namespace IOExpanderBPortAPin
{
}   // namespace IOExpanderBPortAPin

/*
 *  I/O expander B Port B pin namespace.
 */
namespace IOExpanderBPortBPin
{
}   // namespace IOExpanderBPortBPin
}   // namespace firmware
}   // namespace biped

#endif  // COMMON_PIN_H_
