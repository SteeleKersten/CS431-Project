/**
 *  @file   wifi.cpp
 *  @author Simon Yu
 *  @date   09/12/2023
 *  @brief  Wi-Fi class source.
 *
 *  This file implements the Wi-Fi class.
 */

/*
 *  External headers.
 */
#include <WiFi.h>

/*
 *  Project headers.
 */
#include "common/parameter.h"
#include "platform/wifi.h"

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
WiFi::WiFi()
{
}

void
WiFi::initialize()
{
    /*
     *  Initialize and configure the Arduino Wi-Fi driver.
     */
    ::WiFi.begin(NetworkParameter::ssid, NetworkParameter::passphrase);
    ::WiFi.setSleep(false);
}

std::string
WiFi::getWiFiLocalIP() const
{
    /*
     *  Obtain Wi-Fi local IP address from the Arduino Wi-Fi driver,
     *  convert it into C++ STL string, and return.
     */
    return std::string(::WiFi.localIP().toString().c_str());
}

wl_status_t
WiFi::getWiFiStatus() const
{
    /*
     *  Return the Wi-Fi status, defined in wl_status_t
     *  enum in the Arduino WiFiType header, from the Arduino Wi-Fi
     *  driver.
     */
    return ::WiFi.status();
}
}   // namespace firmware
}   // namespace biped
