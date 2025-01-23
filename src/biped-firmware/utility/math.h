/**
 *  @file   math.h
 *  @author Simon Yu
 *  @date   01/18/2022
 *  @brief  Math function header.
 *
 *  This file defines the math functions.
 */

/*
 *  Include guard.
 */
#ifndef UTILITY_MATH_H_
#define UTILITY_MATH_H_

/*
 *  External headers.
 */
#include <freertos/FreeRTOS.h>

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
 *  @param  data Data.
 *  @param  lower_bound Lower bound.
 *  @param  upper_bound Upper bound.
 *  @return Data clamped between the given lower and upper bound.
 *  @brief  Clamp data between the lower and upper bound.
 *
 *  This function clamps the given data between the given
 *  lower and upper bound.
 */
double
clamp(const double& data, const double& lower_bound, const double& upper_bound);

/**
 *  @param  degrees Data in degrees.
 *  @return Data in radians.
 *  @brief  Convert degrees to radians.
 *
 *  This function converts data from degrees to radians.
 */
double
degreesToRadians(const double& degrees);

/**
 *  @param  millimeters Data in millimeters.
 *  @return Data in meters.
 *  @brief  Convert millimeters to meters.
 *
 *  This function converts data from millimeters to meters.
 */
double
millimetersToMeters(const double& millimeters);

/**
 *  @param  microseconds Data in microseconds.
 *  @return Data in seconds.
 *  @brief  Convert microseconds to seconds.
 *
 *  This function converts data from microseconds to seconds.
 */
double
microsecondsToSeconds(const double& microseconds);

/**
 *  @param  milliseconds Data in milliseconds.
 *  @return Data in seconds.
 *  @brief  Convert milliseconds to seconds.
 *
 *  This function converts data from milliseconds to seconds.
 */
double
millisecondsToSeconds(const double& milliseconds);

/**
 *  @param  milliseconds Data in milliseconds.
 *  @return Data in ticks.
 *  @brief  Convert milliseconds to ticks.
 *
 *  This function converts data from milliseconds to ticks.
 */
TickType_t
millisecondsToTicks(const double& milliseconds);

/**
 *  @param  radians Data in radians.
 *  @return Data in degrees.
 *  @brief  Convert radians to degrees.
 *
 *  This function converts data from radians to degrees.
 */
double
radiansToDegrees(const double& radians);

/**
 *  @param  seconds Data in seconds.
 *  @return Data in microseconds.
 *  @brief  Convert seconds to microseconds.
 *
 *  This function converts data from seconds to microseconds.
 */
double
secondsToMicroseconds(const double& seconds);

/**
 *  @param  seconds Data in seconds.
 *  @return Data in milliseconds.
 *  @brief  Convert seconds to milliseconds.
 *
 *  This function converts data from seconds to milliseconds.
 */
double
secondsToMilliseconds(const double& seconds);

/**
 *  @param  ticks Data in ticks.
 *  @return Data in milliseconds.
 *  @brief  Convert ticks to milliseconds.
 *
 *  This function converts data from ticks to milliseconds.
 */
double
ticksToMilliseconds(const TickType_t& ticks);
}   // namespace firmware
}   // namespace biped

#endif  // UTILITY_MATH_H_
