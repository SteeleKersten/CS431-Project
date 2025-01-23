/**
 *  @file   neopixel.cpp
 *  @author Simon Yu
 *  @date   04/01/2022
 *  @brief  NeoPixel class source.
 *
 *  This file implements the NeoPixel class.
 */

/*
 *  Project headers.
 */
#include "common/global.h"
#include "common/parameter.h"
#include "common/pin.h"
#include "platform/neopixel.h"
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
NeoPixel::NeoPixel() : neopixel_(NeoPixelParameter::size, ESP32Pin::neopixel, NEO_GRB + NEO_KHZ800)
{
    /*
     *  Initialize Adafruit NeoPixel driver object.
     */
    neopixel_.begin();

    /*
     *  Clear the NeoPixel LED array and set brightness.
     */
    clear();
    setBrightness(NeoPixelParameter::brightness_max);

    /*
     *  Initialize NeoPixel frame.
     */
    frame_ = std::make_shared<NeoPixel::Frame>();
    frame_->push_back(Eigen::Vector3i(0, 0, 255));
    frame_->push_back(Eigen::Vector3i(0, 0, 255));
    frame_->push_back(Eigen::Vector3i(0, 0, 255));
    frame_->push_back(Eigen::Vector3i(0, 0, 255));

    /*
     *  Initialize NeoPixel error frame.
     */
    frame_error_ = std::make_shared<NeoPixel::Frame>();
    frame_error_->push_back(Eigen::Vector3i(255, 0, 0));
    frame_error_->push_back(Eigen::Vector3i(255, 0, 0));
    frame_error_->push_back(Eigen::Vector3i(255, 0, 0));
    frame_error_->push_back(Eigen::Vector3i(255, 0, 0));
}

void
NeoPixel::setBrightness(const int& brightness)
{
    /*
     *  Clamp the brightness value between the minimum and maximum
     *  LED brightness values.
     */
    neopixel_.setBrightness(
            static_cast<int>(clamp(static_cast<double>(brightness),
                    static_cast<double>(NeoPixelParameter::brightness_min),
                    static_cast<double>(NeoPixelParameter::brightness_max))));
}

void
NeoPixel::setFrame(const std::shared_ptr<Frame> frame)
{
    /*
     *  Set the class member NeoPixel frame shared pointer.
     */
    frame_ = frame;
}

void
NeoPixel::clear()
{
    /*
     *  Clear and flush the NeoPixel LED array.
     */
    neopixel_.clear();
    neopixel_.show();
}

void
NeoPixel::show()
{
    /*
     *  Declare NeoPixel frame shared pointer and set to
     *  null pointer.
     */
    std::shared_ptr<Frame> frame = nullptr;

    /*
     *  Set NeoPixel frame shared pointer to be the error
     *  frame if the current worst log level is worse than
     *  or equal to error.
     */
    if (Serial::getLogLevelWorst() <= LogLevel::error)
    {
        frame = frame_error_;
    }
    else
    {
        frame = frame_;
    }

    /*
     *  Validate NeoPixel frame.
     */
    if (!frame || frame->size() != NeoPixelParameter::size)
    {
        Serial(LogLevel::error) << "Invalid frame.";
        return;
    }

    /*
     *  Set individual NeoPixel LED array pixel NeoPixel frame.
     */
    for (int i = 0; i < NeoPixelParameter::size; i ++)
    {
        neopixel_.setPixelColor(i,
                neopixel_.Color(frame->at(i).x(), frame->at(i).y(), frame->at(i).z()));
    }

    /*
     *  Flush the NeoPixel frame to the NeoPixel LED array.
     */
    neopixel_.show();
}
}   // namespace firmware
}   // namespace biped
