/**
 *  @file   camera.cpp
 *  @author Simon Yu
 *  @date   01/11/2023
 *  @brief  Camera class source.
 *
 *  This file implements the camera class.
 */

/*
 *  External headers.
 */
#include <esp_camera.h>
#include <esp_timer.h>
#include <functional>
#include <img_converters.h>

/*
 *  Project headers.
 */
#include "common/parameter.h"
#include "common/pin.h"
#include "network/udp.h"
#include "platform/camera.h"
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
Camera::Camera()
{
    /*
     *  Declare ESP-IDF camera configuration struct.
     */
    camera_config_t camera_config;

    /*
     *  Populate ESP-IDF camera configuration struct with parameters.
     */
    camera_config.ledc_channel = LEDC_CHANNEL_0;
    camera_config.ledc_timer = LEDC_TIMER_0;
    camera_config.pin_d0 = ESP32Pin::camera_d0;
    camera_config.pin_d1 = ESP32Pin::camera_d1;
    camera_config.pin_d2 = ESP32Pin::camera_d2;
    camera_config.pin_d3 = ESP32Pin::camera_d3;
    camera_config.pin_d4 = ESP32Pin::camera_d4;
    camera_config.pin_d5 = ESP32Pin::camera_d5;
    camera_config.pin_d6 = ESP32Pin::camera_d6;
    camera_config.pin_d7 = ESP32Pin::camera_d7;
    camera_config.pin_xclk = ESP32Pin::camera_xclk;
    camera_config.pin_pclk = ESP32Pin::camera_pclk;
    camera_config.pin_vsync = ESP32Pin::camera_vsync;
    camera_config.pin_href = ESP32Pin::camera_href;
    camera_config.pin_sscb_sda = ESP32Pin::camera_sscb_sda;
    camera_config.pin_sscb_scl = ESP32Pin::camera_sscb_scl;
    camera_config.pin_pwdn = ESP32Pin::camera_pwdn;
    camera_config.pin_reset = ESP32Pin::camera_reset;
    camera_config.xclk_freq_hz = CameraParameter::xclk_frequency;
    camera_config.frame_size = FRAMESIZE_HVGA;
    camera_config.pixel_format = PIXFORMAT_JPEG;
    camera_config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    camera_config.fb_location = CAMERA_FB_IN_DRAM;
    camera_config.jpeg_quality = CameraParameter::jpeg_quality;
    camera_config.fb_count = CameraParameter::frame_buffer_count;

    /*
     *  Initialize the ESP-IDF camera driver and validate the initialization.
     */
    if (esp_camera_init(&camera_config) != ESP_OK)
    {
        Serial(LogLevel::error) << "Failed to initialize camera.";
        return;
    }

    /*
     *  Obtain ESP-IDF camera sensor pointer.
     */
    sensor_t *camera_sensor = esp_camera_sensor_get();

    /*
     *  If the ESP-IDF camera sensor pointer is not a null pointer, using the
     *  ESP-IDF camera sensor pointer, set the camera frame orientations.
     */
    if (camera_sensor)
    {
        camera_sensor->set_hmirror(camera_sensor, 1);
        camera_sensor->set_vflip(camera_sensor, 1);
    }
    else
    {
        Serial(LogLevel::warn) << "Camera sensor missing.";
    }
}

size_t
Camera::SendJPGFrameOverUDP(const std::shared_ptr<UDP> udp, const std::string& ip_remote,
        const uint16_t& port)
{
    /*
     *  Declare variables.
     */
    size_t bytes = 0;
    std::string frame_boundary = NetworkParameter::camera_frame_boundary;
    camera_fb_t *frame_buffer = nullptr;
    size_t frame_packet_count = 0;
    uint8_t *jpg_buffer = nullptr;
    size_t jpg_buffer_size = 0;

    /*
     *  If the given UDP shared pointer is a null pointer, return.
     */
    if (!udp)
    {
        return bytes;
    }

    /*
     *  Obtain the camera frame buffer pointer.
     */
    frame_buffer = esp_camera_fb_get();

    /*
     *  If the obtained camera frame buffer pointer is a null pointer, return.
     */
    if (!frame_buffer)
    {
        return bytes;
    }

    if (frame_buffer->format != PIXFORMAT_JPEG)
    {
        /*
         *  Convert the current frame buffer into JPG and store in the JPG buffer.
         */
        bool converted = frame2jpg(frame_buffer, CameraParameter::jpeg_quality_conversion,
                &jpg_buffer, &jpg_buffer_size);

        /*
         *  Free the frame buffer.
         */
        esp_camera_fb_return(frame_buffer);
        frame_buffer = nullptr;

        /*
         *  If the JPG conversion failed, return.
         */
        if (!converted)
        {
            return bytes;
        }
    }
    else
    {
        /*
         *  Set the JPG frame buffer directly as the frame buffer.
         */
        jpg_buffer = frame_buffer->buf;
        jpg_buffer_size = frame_buffer->len;
    }

    /*
     *  Compute the number of camera frame packets and append it to the camera
     *  frame boundary.
     */
    frame_packet_count = std::ceil(
            static_cast<double>(jpg_buffer_size) / NetworkParameter::buffer_size_camera);
    frame_boundary += static_cast<char>(frame_packet_count);

    /*
     *  Write camera frame boundary packet to UDP with the given remote IP address
     *  and UDP port.
     */
    bytes += udp->write(ip_remote, port, frame_boundary);

    /*
     *  Write the JPG buffer to UDP with the given remote IP address and UDP port.
     */
    bytes += udp->writeBuffer(ip_remote, port, jpg_buffer, jpg_buffer_size);

    /*
     *  Free buffers.
     */
    if (frame_buffer)
    {
        esp_camera_fb_return(frame_buffer);
        frame_buffer = nullptr;
        jpg_buffer = nullptr;
    }
    else if (jpg_buffer)
    {
        free(jpg_buffer);
        jpg_buffer = nullptr;
    }

    /*
     *  Return bytes written.
     */
    return bytes;
}
}   // namespace firmware
}   // namespace biped
