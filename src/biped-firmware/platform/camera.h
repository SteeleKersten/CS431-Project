/**
 *  @file   camera.h
 *  @author Simon Yu
 *  @date   01/11/2023
 *  @brief  Camera class header.
 *
 *  This file defines the camera class.
 */

/*
 *  Include guard.
 */
#ifndef PLATFORM_CAMERA_H_
#define PLATFORM_CAMERA_H_

/*
 *  External headers.
 */
#include <memory>

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
/*
 *  Forward declarations.
 */
class UDP;

/**
 *  @brief  Camera class.
 *
 *  This class provides functions for obtaining camera frames
 *  from the ESP-IDF camera driver and sending them over UDP.
 */
class Camera
{
public:

    /**
     *  @brief  Camera class constructor.
     *
     *  This constructor initializes all class member variables.
     *  Additionally, the constructor initializes and configures the
     *  ESP-IDF camera driver.
     */
    Camera();

    /**
     *  @param  udp UDP object shared pointer.
     *  @param  ip_remote Remote IP address.
     *  @param  port UDP port.
     *  @return Size sent, in bytes.
     *  @brief  Send the camera frame in JPG over the given UDP interface.
     *
     *  This function obtains the camera frame from the ESP-IDF camera
     *  driver and sends the obtained camera frame in JPG over the
     *  given UDP interface.
     */
    size_t
    SendJPGFrameOverUDP(const std::shared_ptr<UDP> udp, const std::string& ip_remote,
            const uint16_t& port);
};
}   // namespace firmware
}   // namespace biped

#endif  // PLATFORM_CAMERA_H_
