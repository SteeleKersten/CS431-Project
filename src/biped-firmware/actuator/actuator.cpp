/**
 *  @file   actuator.cpp
 *  @author Simon Yu
 *  @date   01/13/2022
 *  @brief  Actuator class source.
 *
 *  This file implements the actuator class.
 */

/*
 *  External headers.
 */
#include <Arduino.h>

/*
 *  Project headers.
 */
#include "actuator/actuator.h"
#include "common/global.h"
#include "common/parameter.h"
#include "common/pin.h"
#include "controller/controller.h"
#include "platform/io_expander.h"
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
Actuator::Actuator()
{
    /*
     *  Using the Arduino pinMode function, set pin mode
     *  for the motor PWM pins.
     *
     *  Refer to the pin header for the motor
     *  PWM pins.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */

    /*
     *  Validate I/O expander global shared pointer.
     */
    if (io_expander_a_)
    {
        /*
         *  Using the I/O expander pinModePort functions,
         *  set pin mode for the motor direction and
         *  enable pins.
         *
         *  Refer to the pin header for the motor
         *  direction and enable pins.
         *
         *  TODO LAB 6 YOUR CODE HERE.
         */
    }
    else
    {
        Serial(LogLevel::error) << "IO expander A missing.";
    }
}

ActuationCommand
Actuator::getActuationCommand() const
{
    /*
     *  Return the class member actuation command struct.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
    return ActuationCommand();
}

void
Actuator::actuate(const ActuationCommand& actuation_command)
{
    /*
     *  Store the given actuation command struct in the function
     *  parameter to the class member actuation command struct.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */

    /*
     *  Using the I/O expander digitalWritePort functions, write
     *  motor enable value from the class member actuation command
     *  struct to the motor enable pin.
     *
     *  Refer to the pin header for the motor enable pin.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */

    /*
     *  Using the I/O expander digitalWritePort functions, write
     *  motor direction values from the class member actuation command
     *  struct to the motor direction pins.
     *
     *  Refer to the pin header for the motor direction pins.
     *
     *  Note that Biped is expected to move forward
     *  with both motor direction values being true.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */

    /*
     *  Using the clamp function from the math header, clamp
     *  the motor PWM values from the class member actuation
     *  command struct to be in between the minimum and the
     *  maximum PWM values defined in the parameter header.
     *
     *  Then, using the Arduino write functions, write
     *  the clamped values to the motor PWM pins.
     *
     *  Remember to static_cast the motor parameter values from
     *  the parameter header to doubles before passing them to
     *  the clamp function, and also static_cast the clamped
     *  values back to integers before passing them to the
     *  Arduino write functions.
     *
     *  Note that one should always use C++ explicit type casts
     *  (static_cast, dynamic_cast, etc.) when programming in C++
     *  instead of the C-style type cast, as the C++ type casts are
     *  checked by the compiler, whereas C-style casts are not, and
     *  can fail at runtime.
     *
     *  Learn more about C++ type casting here:
     *  https://cplusplus.com/doc/oldtutorial/typecasting/
     *
     *  Note that the PWM values are analog values. Thus, one should
     *  use the Arduino analogWrite function instead of digitalWrite.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
}
}   // namespace firmware
}   // namespace biped
