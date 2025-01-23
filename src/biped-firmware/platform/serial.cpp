/**
 *  @file   serial.cpp
 *  @author Simon Yu
 *  @date   11/19/2021
 *  @brief  Serial class source.
 *
 *  This file implements the serial class.
 */

/*
 *  External headers.
 */
#include <Arduino.h>

/*
 *  Project headers.
 */
#include "common/parameter.h"
#include "platform/serial.h"

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
Serial::Serial(const LogLevel& log_level, const bool& raw) : log_level_(log_level), raw_(raw)
{
    /*
     *  Do not initialize if the class member log
     *  level is greater than the class member maximum
     *  log level.
     */
    if (log_level_ > log_level_max_)
    {
        return;
    }

    /*
     *  Stream log level tags to the class member string
     *  stream buffer if not in raw mode.
     */
    if (!raw)
    {
        switch (log_level_)
        {
            case LogLevel::fatal:
            {
                ss_ << "[FATAL]: ";
                break;
            }
            case LogLevel::error:
            {
                ss_ << "[ERROR]: ";
                break;
            }
            case LogLevel::warn:
            {
                ss_ << "[WARN]: ";
                break;
            }
            case LogLevel::info:
            {
                ss_ << "[INFO]: ";
                break;
            }
            case LogLevel::debug:
            {
                ss_ << "[DEBUG]: ";
                break;
            }
            case LogLevel::trace:
            {
                ss_ << "[TRACE]: ";
                break;
            }
            default:
            {
                ss_ << "[FATAL]: ";
                break;
            }
        }
    }
}

Serial::~Serial()
{
    /*
     *  Do not flush the string stream buffer if
     *  the class member log level is greater than
     *  the class member maximum log level.
     */
    if (log_level_ > log_level_max_)
    {
        return;
    }

    /*
     *  Flush the string stream buffer to the serial
     *  connection based on the printing mode.
     */
    if (raw_)
    {
        ::Serial.print(ss_.str().c_str());
    }
    else
    {
        ::Serial.println(ss_.str().c_str());
    }

    /*
     *  Update the current worst log level.
     */
    if (log_level_ < log_level_worst_)
    {
        log_level_worst_ = log_level_;
    }
}

void
Serial::initialize()
{
    /*
     *  Initialize Arduino serial driver object.
     */
    ::Serial.begin(SerialParameter::baud_rate);
}

LogLevel
Serial::getLogLevelWorst()
{
    /*
     *  Return the current worst log level.
     */
    return log_level_worst_;
}

void
Serial::setLogLevelMax(const LogLevel& log_level_max)
{
    /*
     *  Set the class member maximum log level.
     */
    log_level_max_ = log_level_max;
}

Serial&
Serial::operator<<(const StreamManipulator& item)
{
    /*
     *  Do not stream the data if the class member
     *  log level is greater than the class member
     *  maximum log level.
     */
    if (log_level_ > log_level_max_)
    {
        return *this;
    }

    /*
     *  Interpret the given stream manipulator and perform
     *  their respective functionalities.
     */
    switch (item)
    {
        case StreamManipulator::carriage_return:
        {
            ss_ << "\r";
            break;
        }
        case StreamManipulator::endl:
        {
            ss_ << "\r\n";
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

/*
 *  Initialize static class member variables.
 */
LogLevel Serial::log_level_max_ = LogLevel::trace;
LogLevel Serial::log_level_worst_ = LogLevel::trace;
}   // namespace firmware
}   // namespace biped
