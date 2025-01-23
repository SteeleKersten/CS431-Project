/**
 *  @file   pid_controller.cpp
 *  @author Simon Yu
 *  @date   01/13/2022
 *  @brief  PID controller class source.
 *
 *  This file implements the PID controller class.
 */

/*
 *  Project headers.
 */
#include "common/parameter.h"
#include "controller/pid_controller.h"
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
PIDController::PIDController() : state_(0), reference_(0), period_(0), error_differential_(0),
        error_integral_(0)
{
}

double
PIDController::getReference() const
{
    /*
     *  Return the class member PID controller reference (R).
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
    return 0;
}

void
PIDController::setGain(const PIDControllerGain& gain)
{
    /*
     *  Set the class member PID controller gain.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  The existing integrated error (integral of e)
     *  becomes meaningless with new gains. Reset the
     *  integrated error.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
}

void
PIDController::setSaturation(const ControllerSaturation& saturation)
{
    /*
     *  Set the class member PID controller saturation.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
}

void
PIDController::setState(const double& state)
{
    /*
     *  Set the class member plant state input (Y).
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
}

void
PIDController::setReference(const double& reference)
{
    /*
     *  Set the PID controller reference (R).
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  The existing integrated error (integral of e)
     *  becomes meaningless with new references. Reset the
     *  integrated error.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
}

void
PIDController::setPeriod(const double& period)
{
    /*
     *  Set the class member PID controller period.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  The existing integrated error (integral of e)
     *  becomes meaningless with a new period. Reset the
     *  integrated error.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
}

void
PIDController::setErrorDifferential(const double& error_differential)
{
    /*
     *  Set the class member error derivative input (delta e).
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
}

void
PIDController::resetErrorIntegral()
{
    /*
     *  Reset the class member integrated error (integral of e) to 0.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
}

double
PIDController::control()
{
    /*
     *  Validate PID controller period.
     */
    if (period_ <= 0)
    {
        Serial(LogLevel::error) << "Invalid period.";
        return 0;
    }

    /*
     *  Calculate the current error (e) between the plant input
     *  state (Y) and PID controller reference (R). Using the clamp
     *  function in the math header, clamp the calculated error
     *  between the input saturation upper and lower bounds.
     *
     *  For the "correct" signedness, calculate the error (e) by
     *  subtracting state with reference (Y - R). Otherwise, the signs
     *  of your PID controller gains may be flipped.
     *
     *  Refer to Lecture 13 for the definition of the PID controller.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Calculate the new discrete integral of error (integral of e).
     *  Using the clamp function in the math header, clamp the
     *  calculated integral of error (integral of e) between the
     *  negative and positive maximum integrated error from the
     *  class member PID controller gain struct.
     *
     *  The new discrete integrated error, i.e., the class member integral
     *  error variable, is the current integrated error, i.e., also
     *  the class member integral error variable, plus the product between
     *  the PID controller period and the current error (e).
     *
     *  The above is essentially the Riemann sum. Learn more here:
     *  https://en.wikipedia.org/wiki/Riemann_sum
     *
     *  Refer to Lecture 13 for the definition of the PID controller.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Calculate the proportional output using the current error (e).
     *
     *  Refer to Lecture 13 for the definition of the PID controller.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Calculate the integral output using the new discrete integral
     *  of error (integral of e).
     *
     *  Refer to Lecture 13 for the definition of the PID controller.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Calculate the differential output using the class member
     *  differential error variable.
     *
     *  In theory, the differential output is proportional
     *  to the time derivative of the error (e). Instead of
     *  calculating the derivative of the error (e), we
     *  directly measure the derivative using hardware
     *  sensors for better performance. For example, for
     *  position control, the differential output would be
     *  proportional to the derivative of the error (e),
     *  which, in this example, is simply velocity (assuming
     *  the controller reference remains unchanged over
     *  time). Instead of calculating the velocity by taking
     *  the derivative of the position, we directly measure
     *  it using hardware sensors (i.e., encoders).
     *
     *  Refer to Lecture 13 for the definition of the PID controller.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Sum up all of the above proportional, integral, and
     *  differential output.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Using the clamp function in the math header, return the
     *  sum of the output computed above clamped between the output
     *  saturation upper and lower bounds as the final output
     *  of the PID controller.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
    return 0;
}
}   // namespace firmware
}   // namespace biped
