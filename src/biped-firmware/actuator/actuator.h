/**
 *  @file   actuator.h
 *  @author Simon Yu
 *  @date   01/13/2022
 *  @brief  Actuator class header.
 *
 *  This file defines the actuator class.
 */

/*
 *  Include guard.
 */
#ifndef ACTUATOR_ACTUATOR_H_
#define ACTUATOR_ACTUATOR_H_

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
 *  @brief  Actuator class.
 *
 *  This class provides functions for controlling actuators,
 *  i.e., the motors.
 */
class Actuator
{
public:

    /**
     *  @brief  Actuator class constructor.
     *
     *  This constructor initializes all class member variables.
     *  Additionally, the constructor sets the related I/O pin
     *  modes.
     */
    Actuator();

    /**
     *  @return Actuation command struct.
     *  @brief  Get the class member actuation command struct.
     *
     *  This function returns the class member actuation command struct.
     */
    ActuationCommand
    getActuationCommand() const;

    /**
     *  @param  actuation_command actuation command struct.
     *  @brief  Actuation function.
     *
     *  This function takes an actuation command struct
     *  and performs actuation. This function is expected
     *  to be called periodically.
     */
    void
    actuate(const ActuationCommand& actuation_command);

private:

    ActuationCommand actuation_command_;    //!< Actuation command struct.
};
}   // namespace firmware
}   // namespace biped

#endif  // ACTUATOR_ACTUATOR_H_
