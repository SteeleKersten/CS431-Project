/**
 *  @file   udp.h
 *  @author Simon Yu
 *  @date   10/30/2023
 *  @brief  UDP class header.
 *
 *  This file defines the UDP class.
 */

/*
 *  Include guard.
 */
#ifndef NETWORK_UDP_H_
#define NETWORK_UDP_H_

/*
 *  External headers.
 */
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <WiFiUdp.h>

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
 *  @brief  UDP class.
 *
 *  This class provides functions for reading from and
 *  writing to a UDP port.
 */
class UDP
{
public:

    /**
     *  @brief  UDP class constructor.
     *
     *  This constructor instantiates the UDP class.
     */
    UDP();

    /**
     *  @param  port UDP port.
     *  @brief  Initialize UDP.
     *
     *  This function initializes the Arduino Wi-Fi UDP
     *  driver with the given port.
     */
    void
    initialize(const uint16_t& port);

    /**
     *  @param  ip_remote Remote IP address.
     *  @param  port UDP port.
     *  @param  size Read size, in bytes.
     *  @return String read.
     *  @brief  UDP string reading function.
     *
     *  This function reads a string of the given size from
     *  the given remote IP address and port.
     */
    std::string
    read(const std::string& ip_remote, const uint16_t& port, const size_t& size);

    /**
     *  @param  ip_remote Remote IP address.
     *  @param  port UDP port.
     *  @param  size Read size, in bytes.
     *  @return Shared pointer to the buffer read.
     *  @brief  UDP buffer reading function.
     *
     *  This function reads a buffer of the given size from
     *  the given remote IP address and port.
     */
    std::shared_ptr<std::vector<char>>
    readBuffer(const std::string& ip_remote, const uint16_t& port, const size_t& size);

    /**
     *  @param  ip_remote Remote IP address.
     *  @param  port UDP port.
     *  @param  data Data as a string.
     *  @return Size written, in bytes.
     *  @brief  UDP string writing function.
     *
     *  This function writes a given string to the given remote
     *  IP address and port.
     */
    size_t
    write(const std::string& ip_remote, const uint16_t& port, const std::string& data);

    /**
     *  @param  ip_remote Remote IP address.
     *  @param  port UDP port.
     *  @param  buffer Buffer pointer.
     *  @param  size Size of the buffer, in bytes.
     *  @return Size written, in bytes.
     *  @brief  UDP buffer writing function.
     *
     *  This function writes a given buffer with the given size to
     *  the given remote IP address and port.
     */
    size_t
    writeBuffer(const std::string& ip_remote, const uint16_t& port, const uint8_t* buffer,
            const size_t& size);

private:

    WiFiUDP udp_;   //!< Arduino Wi-Fi UDP driver object.
    std::mutex mutex_udp_read_; //!< UDP reading mutex.
    std::mutex mutex_udp_write_;    //!< UDP writing mutex.
};
}   // namespace firmware
}   // namespace biped

#endif  // NETWORK_UDP_H_
