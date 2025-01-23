/**
 *  @file   display.cpp
 *  @author Simon Yu
 *  @date   01/02/2023
 *  @brief  Display class source.
 *
 *  This file implements the display class.
 */

/*
 *  Project headers.
 */
#include "common/global.h"
#include "common/parameter.h"
#include "platform/display.h"
#include "sensor/sensor.h"

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
Display::Display(const unsigned& line, const bool& raw) : line_(line)
{
    /*
     *  Acquire the Adafruit display driver mutex.
     */
    lock_sh1107_.lock();

    /*
     *  Set up the display for printing, depending on the printing mode.
     */
    if (raw)
    {
        restoreCursor();
    }
    else
    {
        clearLine();
        carriageReturn();
    }
}

Display::~Display()
{
    /*
     *  Flush the class member string stream buffer to the Adafruit
     *  display driver.
     */
    flush();

    /*
     *  Save the current X cursor position and unmark the display flag.
     */
    cursor_x_locations_[line_] = sh1107_.getCursorX();
    displayed_ = false;

    /*
     *  Release the Adafruit display driver mutex.
     */
    lock_sh1107_.unlock();
}

void
Display::initialize()
{
    /*
     *  Declare text bounds
     */
    int16_t text_bound_x = 0;
    int16_t text_bound_y = 0;

    /*
     *  Initialize and configure the Adafruit display driver object.
     */
    sh1107_.begin(AddressParameter::display, false);
    sh1107_.setCursor(0, 0);
    sh1107_.setRotation(DisplayParameter::rotation_upright);
    sh1107_.setTextSize(DisplayParameter::text_size);
    sh1107_.setTextColor(SH110X_WHITE);
    sh1107_.setTextWrap(false);
    sh1107_.getTextBounds(" ", 0, 0, &text_bound_x, &text_bound_y, &font_width_, &font_height_);
    sh1107_.clearDisplay();
    sh1107_.display();

    /*
     *  Configure Z acceleration low-pass filter.
     */
    low_pass_filter_acceleration_z_.setBeta(DisplayParameter::low_pass_filter_beta);
}

void
Display::display()
{
    if (sensor_)
    {
        /*
         *  Get low-pass filtered Z acceleration from IMU.
         */
        static bool inverted = false;
        const auto acceleration_z = low_pass_filter_acceleration_z_.filter(
                sensor_->getIMUData().acceleration_z);

        /*
         *  Set display orientation based on filtered Z acceleration.
         */
        if (!inverted && acceleration_z < DisplayParameter::acceleration_z_rotation)
        {
            sh1107_.setRotation(DisplayParameter::rotation_inverted);
            sh1107_.clearDisplay();
            inverted = true;
        }
        else if (inverted && acceleration_z >= DisplayParameter::acceleration_z_rotation)
        {
            sh1107_.setRotation(DisplayParameter::rotation_upright);
            sh1107_.clearDisplay();
            inverted = false;
        }
    }

    /*
     *  Flush display driver buffer to the OLED display and mark
     *  the display flag.
     */
    if (!displayed_)
    {
        sh1107_.display();
        displayed_ = true;
    }
}

Display&
Display::operator<<(const StreamManipulator& item)
{
    /*
     *  Interpret the given stream manipulator and perform
     *  their respective functionalities.
     */
    switch (item)
    {
        case StreamManipulator::carriage_return:
        {
            flush();
            carriageReturn();
            break;
        }
        case StreamManipulator::clear_line:
        {
            flush();
            clearLine();
            break;
        }
        case StreamManipulator::endl:
        {
            ss_ << "\n";
            line_ ++;
            break;
        }
        default:
        {
            break;
        }
    }

    /*
     *  Return a reference to this object.
     */
    return *this;
}

void
Display::carriageReturn()
{
    /*
     *  Insert carriage return by setting X cursor position to zero.
     */
    sh1107_.setCursor(0, line_ * font_height_);
}

void
Display::clearLine()
{
    /*
     *  Clear current line by filling the line with a black rectangle.
     */
    sh1107_.fillRect(0, line_ * font_height_, DisplayParameter::width, font_height_, 0);
}

void
Display::flush()
{
    /*
     *  Declare text bounds and dimensions.
     */
    int16_t text_bound_x = 0;
    int16_t text_bound_y = 0;
    uint16_t text_height = 0;
    uint16_t text_width = 0;

    /*
     *  Get text bounds in the string stream buffer, fill
     *  the bounds with a black rectangle, flush the string
     *  stream buffer to the Adafruit display driver buffer,
     *  and clear the string stream buffer.
     */
    sh1107_.getTextBounds(ss_.str().c_str(), sh1107_.getCursorX(), sh1107_.getCursorY(),
            &text_bound_x, &text_bound_y, &text_width, &text_height);
    sh1107_.fillRect(sh1107_.getCursorX(), sh1107_.getCursorY(), text_width, text_height, 0);
    sh1107_.print(ss_.str().c_str());
    ss_.str(std::string());
}

void
Display::restoreCursor()
{
    /*
     *  Restore cursor position to the last save position.
     */
    if (cursor_x_locations_.count(line_))
    {
        sh1107_.setCursor(cursor_x_locations_[line_], line_ * font_height_);
    }
    else
    {
        sh1107_.setCursor(0, line_ * font_height_);
    }
}

/*
 *  Initialize static class member variables.
 */
std::map<unsigned, int16_t> Display::cursor_x_locations_ = std::map<unsigned, int16_t>();
volatile bool Display::displayed_ = true;
uint16_t Display::font_height_ = 0;
uint16_t Display::font_width_ = 0;
std::mutex Display::mutex_sh1107_;
std::unique_lock<std::mutex> Display::lock_sh1107_ = std::unique_lock<std::mutex>(
        Display::mutex_sh1107_, std::defer_lock);
LowPassFilter<double> Display::low_pass_filter_acceleration_z_ = LowPassFilter<double>();
Adafruit_SH1107 Display::sh1107_ = Adafruit_SH1107(DisplayParameter::height,
        DisplayParameter::width, &Wire, DisplayParameter::reset_pin, DisplayParameter::preclk,
        DisplayParameter::postclk);
}   // namespace firmware
}   // namespace biped
