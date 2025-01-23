/**
 *  @file   wifi.h
 *  @author Simon Yu
 *  @date   09/12/2023
 *  @brief  Wi-Fi class header.
 *
 *  This file defines the Wi-Fi class.
 */

/*
 *  Include guard.
 */
#ifndef PLATFORM_WIFI_H_
#define PLATFORM_WIFI_H_

/*
 *  External headers.
 */
#include <string>
#include <WiFiType.h>

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
 *  @brief  Wi-Fi class.
 *
 *  This class provides functions for initializing the Arduino
 *  Wi-Fi driver and obtaining Wi-Fi connection status and local
 *  IP address.
 */
class WiFi
{
public:

    /**
     *  @brief  Wi-Fi class constructor.
     *
     *  This constructor instantiates the Wi-Fi object.
     */
    WiFi();

    /**
     *  @brief  Initialize Wi-Fi driver.
     *
     *  This function initializes and configures the Arduino
     *  Wi-Fi driver.
     */
    void
    initialize();

    /**
     *  @return Wi-Fi local IP address.
     *  @brief  Get the Wi-Fi local IP address.
     *
     *  This function returns the local IP address from the Arduino
     *  Wi-Fi driver.
     */
    std::string
    getWiFiLocalIP() const;

    /**
     *  @return Wi-Fi status.
     *  @brief  Get the Wi-Fi status.
     *
     *  This function returns the Wi-Fi status defined in wl_status_t
     *  enum in the Arduino WiFiType header from the Arduino Wi-Fi
     *  driver.
     */
    wl_status_t
    getWiFiStatus() const;
};
}   // namespace firmware
}   // namespace biped

#endif  // PLATFORM_WIFI_H_
