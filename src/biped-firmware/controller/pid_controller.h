/**
 *  @file   pid_controller.h
 *  @author Simon Yu
 *  @date   01/13/2022
 *  @brief  PID controller class header.
 *
 *  This file defines the PID controller class.
 */

/*
 *  Include guard.
 */
#ifndef CONTROLLER_PID_CONTROLLER_H_
#define CONTROLLER_PID_CONTROLLER_H_

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
 *  @brief  PID controller class.
 *
 *  This class provides functions for creating
 *  a PID controller, which generates an output that
 *  minimizes the error between the plant state input
 *  (Y) and a reference (R) in a closed loop.
 *
 *  Refer to Lecture 13 for the definition of the PID
 *  controller.
 */
class PIDController
{
public:

    /**
     *  @brief  PID controller class constructor.
     *
     *  This constructor initializes all class member variables.
     */
    PIDController();

    /**
     *  @return PID controller reference (R).
     *  @brief  Get the PID controller reference (R).
     *
     *  This function returns the PID controller reference (R).
     *
     *  Refer to Lecture 13 for the variable definitions.
     */
    double
    getReference() const;

    /**
     *  @param  gain PID controller gain struct.
     *  @brief  Set the PID controller gain.
     *
     *  This function sets the PID controller gain
     *  and resets the integrated error.
     */
    void
    setGain(const PIDControllerGain& gain);

    /**
     *  @param  saturation Controller saturation struct.
     *  @brief  Set the PID controller saturation.
     *
     *  This function sets the PID controller saturation.
     */
    void
    setSaturation(const ControllerSaturation& saturation);

    /**
     *  @param  state Plant state input (Y).
     *  @brief  Set the Plant state input (Y).
     *
     *  This function sets the plant state input (Y).
     *
     *  Refer to Lecture 13 for the variable definitions.
     */
    void
    setState(const double& state);

    /**
     *  @param  reference PID controller reference (R).
     *  @brief  Set the PID controller reference (R).
     *
     *  This function sets the PID controller reference (R).
     *
     *  Refer to Lecture 13 for the variable definitions.
     */
    void
    setReference(const double& reference);

    /**
     *  @param  period PID controller period, in seconds.
     *  @brief  Set the PID controller period.
     *
     *  This function sets the PID controller period.
     */
    void
    setPeriod(const double& period);

    /**
     *  @param  error_differential Error derivative input (delta e).
     *  @brief  Set the error derivative input (delta e).
     *
     *  This function sets the error derivative input (delta e).
     *
     *  Refer to Lecture 13 for the variable definitions.
     */
    void
    setErrorDifferential(const double& error_differential);

    /**
     *  @brief  Reset the integrated error (integral of e).
     *
     *  This function resets the integrated error (integral of e).
     *
     *  Refer to Lecture 13 for the variable definitions.
     */
    void
    resetErrorIntegral();

    /**
     *  @return Output (u).
     *  @brief  Execute the PID controller.
     *
     *  This function executes the PID controller, which generates control
     *  output (u). This function is expected to be called periodically.
     *
     *  Refer to Lecture 13 for the variable definitions.
     */
    double
    control();

private:

    PIDControllerGain gain_; //!< PID controller gain struct.
    ControllerSaturation saturation_; //!< Controller saturation struct.
    double state_;   //!< Plant state input (Y).
    double reference_;  //!< PID controller reference (R).
    double period_;  //!< PID controller period, in seconds.
    double error_differential_;  //!< Error derivative input (delta e).
    double error_integral_;  //!< Integrated error (integral of e).
};
}   // namespace firmware
}   // namespace biped

#endif  // CONTROLLER_PID_CONTROLLER_H_
