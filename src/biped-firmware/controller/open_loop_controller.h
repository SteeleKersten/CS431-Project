/**
 *  @file   open_loop_controller.h
 *  @author Simon Yu
 *  @date   01/09/2023
 *  @brief  Open-loop controller class header.
 *
 *  This file defines the open-loop controller class.
 */

/*
 *  Include guard.
 */
#ifndef CONTROLLER_OPEN_LOOP_CONTROLLER_H_
#define CONTROLLER_OPEN_LOOP_CONTROLLER_H_

/*
 *  Project headers.
 */
#include "common/parameter.h"
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
 *  @brief  Open-loop controller class.
 *
 *  This class provides functions for creating
 *  an open-loop controller.
 *
 *  Learn more about open-loop controllers here:
 *  https://en.wikipedia.org/wiki/Open-loop_controller
 */
class OpenLoopController
{
public:

    /**
     *  @brief  Open-loop controller class constructor.
     *
     *  This constructor initializes all class member variables.
     */
    OpenLoopController();

    /**
     *  @return Open-loop controller reference (R).
     *  @brief  Get the open-loop controller reference (R).
     *
     *  This function returns the open-loop controller reference (R).
     *
     *  Refer to Lecture 13 for the variable definitions.
     */
    double
    getReference() const;

    /**
     *  @param  gain Open-loop controller gain.
     *  @brief  Set the open-loop controller gain.
     *
     *  This function sets the open-loop controller gain.
     */
    void
    setGain(const double& gain);

    /**
     *  @param  saturation Controller saturation struct.
     *  @brief  Set the open-loop controller saturation.
     *
     *  This function sets the open-loop controller saturation.
     */
    void
    setSaturation(const ControllerSaturation& saturation);

    /**
     *  @param  reference Open-loop controller reference (R).
     *  @brief  Set the open-loop controller reference (R).
     *
     *  This function sets the open-loop controller reference (R).
     *
     *  Refer to Lecture 13 for the variable definitions.
     */
    void
    setReference(const double& reference);

    /**
     *  @return Output (u).
     *  @brief  Execute the open-loop controller.
     *
     *  This function executes the open-loop controller, which generates control
     *  output (u). This function is expected to be called periodically.
     *
     *  Refer to Lecture 13 for the variable definitions.
     */
    double
    control();

private:

    double gain_; //!< Open-loop controller gain.
    double reference_;  //!< Open-loop controller reference (R).
    ControllerSaturation saturation_; //!< Controller saturation struct.
};
}   // namespace firmware
}   // namespace biped

#endif  // CONTROLLER_OPEN_LOOP_CONTROLLER_H_
