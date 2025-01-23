/**
 *  @file   neopixel.h
 *  @author Simon Yu
 *  @date   04/01/2022
 *  @brief  NeoPixel class header.
 *
 *  This file defines the NeoPixel class.
 */

/*
 *  Include guard.
 */
#ifndef PLATFORM_NEOPIXEL_H_
#define PLATFORM_NEOPIXEL_H_

/*
 *  External headers.
 */
#include <Adafruit_NeoPixel.h>
#include <ArduinoEigenDense.h>
#include <memory>
#include <vector>

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
 *  @brief  NeoPixel class.
 *
 *  This class provides functions for controlling the NeoPixel
 *  LED array.
 */
class NeoPixel
{
public:

    using Frame = std::vector<Eigen::Vector3i>; //!< NeoPixel frame type.

    /**
     *  @brief  NeoPixel class constructor.
     *
     *  This constructor initializes all class member variables.
     *  Additionally, the constructor initializes the NeoPixel
     *  driver and the NeoPixel frames.
     */
    NeoPixel();

    /**
     *  @param  brightness LED brightness.
     *  @brief  Set the brightness of the NeoPixel LED array.
     *
     *  This function sets the brightness of the NeoPixel
     *  LED array. The brightness value is clamped between
     *  the minimum and maximum LED brightness values.
     */
    void
    setBrightness(const int& brightness);

    /**
     *  @param  frame NeoPixel frame shared pointer.
     *  @brief  Set the NeoPixel frame.
     *
     *  This function sets the NeoPixel frame.
     */
    void
    setFrame(const std::shared_ptr<Frame> frame);

    /**
     *  @brief  Clear the NeoPixel LED array.
     *
     *  This function clears the NeoPixel LED array.
     */
    void
    clear();

    /**
     *  @brief  Flush the NeoPixel frames to the NeoPixel LED array.
     *
     *  This function flushes the class member NeoPixel frames to the
     *  NeoPixel LED array. The frame flushed depends on the current
     *  worst log level. This function is expected to be called
     *  periodically.
     */
    void
    show();

private:

    std::shared_ptr<Frame> frame_;  //!< NeoPixel frame shared pointer.
    std::shared_ptr<Frame> frame_error_;    //!< NeoPixel error frame shared pointer.
    Adafruit_NeoPixel neopixel_;    //!< Adafruit NeoPixel driver object.
};
}   // namespace firmware
}   // namespace biped

#endif  // PLATFORM_NEOPIXEL_H_
