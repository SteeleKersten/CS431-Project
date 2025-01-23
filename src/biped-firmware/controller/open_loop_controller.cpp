/**
 *  @file   open_loop_controller.cpp
 *  @author Simon Yu
 *  @date   01/09/2023
 *  @brief  Open-loop controller class source.
 *
 *  This file implements the open-loop controller class.
 */

/*
 *  Project headers.
 */
#include "common/parameter.h"
#include "controller/open_loop_controller.h"
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
OpenLoopController::OpenLoopController() : gain_(0), reference_(0)
{
}

double
OpenLoopController::getReference() const
{
    /*
     *  Return the class member open-loop controller reference (R).
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
    return 0;
}

void
OpenLoopController::setGain(const double& gain)
{
    /*
     *  Set the class member open-loop controller gain.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
}

void
OpenLoopController::setSaturation(const ControllerSaturation& saturation)
{
    /*
     *  Set the class member open-loop controller saturation.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
}

void
OpenLoopController::setReference(const double& reference)
{
    /*
     *  Set the class member open-loop controller reference (R).
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
}

double
OpenLoopController::control()
{
    /*
     *  Calculate the open-loop controller output.
     *
     *  The open-loop controller output is the product between
     *  the open-loop controller gain and the clamped
     *  open-loop controller reference (R) between the input
     *  saturation upper and lower bounds. Use the clamp function
     *  in the math header.
     *
     *  Learn more about open-loop controllers here:
     *  https://en.wikipedia.org/wiki/Open-loop_controller
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Return the output computed above clamped between the
     *  output saturation upper and lower bounds as the final
     *  output of the open-loop controller. Use the
     *  clamp function in the math header.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
    return 0;
}
}   // namespace firmware
}   // namespace biped
