/**
 *  @file   time_of_flight.h
 *  @author Simon Yu
 *  @date   01/10/2023
 *  @brief  Time-of-flight class header.
 *
 *  This file defines the time-of-flight class.
 */

/*
 *  Include guard.
 */
#ifndef PLATFORM_TIME_OF_FLIGHT_H_
#define PLATFORM_TIME_OF_FLIGHT_H_

/*
 *  External headers.
 */
#include <vl53l4cx_class.h>

/*
 *  Project headers.
 */
#include "common/type.h"
#include "platform/io_expander.h"
#include "utility/median_filter.hpp"

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
 *  @brief  Time-of-flight class.
 *
 *  This class provides functions for reading from the time-of-flight
 *  sensors.
 */
class TimeOfFlight
{
public:

    /**
     *  @param  address I2C address.
     *  @param  shutdown_pin Shutdown pin.
     *  @param  io_expander I/O expander shared pointer.
     *  @brief  Time-of-flight class constructor.
     *
     *  This constructor initializes all class member variables and
     *  the time-of-flight sensor driver.
     */
    TimeOfFlight(const uint8_t& address, const uint8_t& shutdown_pin,
            const std::shared_ptr<IOExpander> io_expander);

    /**
     *  @return Time-of-flight sensor ranging data.
     *  @brief  Time-of-flight sensor reading function.
     *
     *  This function reads the time-of-flight sensor and returns the
     *  minimum ranging data. The function also filters the ranging data
     *  using the median filter.
     */
    double
    read();

private:

    MedianFilter<double> median_filter_;    //!< Median filter object.
    VL53L4CX vl53l4cx_; //!< VL53L4CX time-of-flight sensor driver object.

};
}   // namespace firmware
}   // namespace biped

#endif  // PLATFORM_TIME_OF_FLIGHT_H_
