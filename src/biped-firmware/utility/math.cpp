/**
 *  @file   math.cpp
 *  @author Simon Yu
 *  @date   01/18/2022
 *  @brief  Math function source.
 *
 *  This file implements the math functions.
 */

/*
 *  External headers.
 */
#include <cmath>

/*
 *  Project headers.
 */
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
double
clamp(const double& data, const double& lower_bound, const double& upper_bound)
{
    /*
     *  Clamp the given data between the given lower and upper bound.
     */
    return data < lower_bound ? lower_bound : (data > upper_bound ? upper_bound : data);
}

double
degreesToRadians(const double& degrees)
{
    /*
     *  Convert degrees to radians.
     */
    return degrees / 180 * M_PI;
}

double
millimetersToMeters(const double& millimeters)
{
    /*
     *  Convert millimeters to meters.
     */
    return millimeters / 1000;
}

double
microsecondsToSeconds(const double& microseconds)
{
    /*
     *  Convert microseconds to seconds.
     */
    return microseconds / 1000 / 1000;
}

double
millisecondsToSeconds(const double& milliseconds)
{
    /*
     *  Convert milliseconds to seconds.
     */
    return milliseconds / 1000;
}

TickType_t
millisecondsToTicks(const double& milliseconds)
{
    /*
     *  Convert milliseconds to ticks.
     */
    return pdMS_TO_TICKS(milliseconds);
}

double
radiansToDegrees(const double& radians)
{
    /*
     *  Convert radians to degrees.
     */
    return radians / M_PI * 180;
}

double
secondsToMicroseconds(const double& seconds)
{
    /*
     *  Convert seconds to microseconds.
     */
    return secondsToMilliseconds(seconds) * 1000;
}

double
secondsToMilliseconds(const double& seconds)
{
    /*
     *  Convert seconds to milliseconds.
     */
    return seconds * 1000;
}

double
ticksToMilliseconds(const TickType_t& ticks)
{
    /*
     *  Convert ticks to milliseconds.
     */
    return ticks * portTICK_PERIOD_MS;
}
}   // namespace firmware
}   // namespace biped
