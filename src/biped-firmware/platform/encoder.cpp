/**
 *  @file   encoder.cpp
 *  @author Simon Yu
 *  @date   01/05/2023
 *  @brief  Encoder class source.
 *
 *  This file implements the encoder class.
 */

/*
 *  External headers.
 */
#include <Arduino.h>
#include <esp32-hal-gpio.h>

/*
 *  Project headers.
 */
#include "common/global.h"
#include "common/parameter.h"
#include "common/pin.h"
#include "platform/encoder.h"
#include "platform/serial.h"
#include "task/interrupt.h"

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
Encoder::Encoder() : steps_left_(0), steps_right_(0)
{
    /*
     *  Using the Arduino pinMode function, set pin mode for
     *  the motor encoder pins. Use pull-up if the pin
     *  mode is input.
     *
     *  Learn more about pull-up resistors here:
     *  https://en.wikipedia.org/wiki/Pull-up_resistor
     *
     *  Refer to the pin header for the motor encoder pins and
     *  the esp32-hal-gpio header for the available pin modes
     *  (GPIO FUNCTIONS).
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */

    /*
     *  Configure X velocity low-pass filter.
     */
    low_pass_filter_velocity_x_.setBeta(EncoderParameter::low_pass_filter_beta);
}

EncoderData
Encoder::getData() const
{
    /*
     *  Return the class member encoder data struct.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
    return EncoderData();
}

void
Encoder::read()
{
    /*
     *  Set the left and right encoder steps in the class
     *  member encoder data struct to be the class member
     *  left and right encoder steps.
     *
     *  Set the overall encoder steps in the class member
     *  encoder data struct to be the average between the
     *  left and right encoder steps.
     *
     *  Set the X position in the class member encoder data
     *  struct by converting the overall encoder steps into
     *  meters using the encoder parameters in the parameter
     *  header.
     *
     *  Refer to the type header for the encoder data struct
     *  and the parameter header for the encoder parameter.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
}

void
Encoder::calculateVelocity()
{
    /*
     *  Declare last overall encoder steps and set to 0.
     */
    static long steps_last = 0;

    /*
     *  Read encoders.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */

    /*
     *  Using the current overall encoder steps in the class member
     *  encoder data struct, calculate the step changes since the last
     *  overall encoder steps, convert the steps changes into meters using
     *  the encoder parameters in the parameter header, calculate the X
     *  velocity using the slow domain period, filter the calculated X velocity
     *  using the low-pass filter, and set the X velocity in the class member
     *  encoder data struct to the calculated X velocity.
     *
     *  Refer to the type header for the encoder data struct, the low-pass filter
     *  header for the low-pass filter functions, and the parameter header for the
     *  encoder and period parameters.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */

    /*
     *  Update the last overall encoder steps local variable to be the current
     *  overall encoder steps in the class member encoder data struct.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
}

void IRAM_ATTR
Encoder::onLeftA()
{
    /*
     *  Using the digitalReadFromISR function from the interrupt header, read the
     *  left encoder pins and update the left encoder steps based on the pins read.
     *
     *  Learn more about the incremental rotary encoder here:
     *  https://lastminuteengineers.com/rotary-encoder-arduino-tutorial/
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
}

void IRAM_ATTR
Encoder::onLeftB()
{
    /*
     *  Using the digitalReadFromISR function from the interrupt header, read the
     *  left encoder pins and update the left encoder steps based on the pins read.
     *
     *  Learn more about the incremental rotary encoder here:
     *  https://lastminuteengineers.com/rotary-encoder-arduino-tutorial/
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
}

void IRAM_ATTR
Encoder::onRightA()
{
    /*
     *  Using the digitalReadFromISR function from the interrupt header, read the
     *  right encoder pins and update the right encoder steps based on the pins read.
     *
     *  Learn more about the incremental rotary encoder here:
     *  https://lastminuteengineers.com/rotary-encoder-arduino-tutorial/
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
}

void IRAM_ATTR
Encoder::onRightB()
{
    /*
     *  Using the digitalReadFromISR function from the interrupt header, read the
     *  right encoder pins and update the right encoder steps based on the pins read.
     *
     *  Learn more about the incremental rotary encoder here:
     *  https://lastminuteengineers.com/rotary-encoder-arduino-tutorial/
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
}
}   // namespace firmware
}   // namespace biped
