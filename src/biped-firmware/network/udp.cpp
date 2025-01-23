/**
 *  @file   udp.cpp
 *  @author Simon Yu
 *  @date   09/12/2023
 *  @brief  UDP class source.
 *
 *  This file implements the UDP class.
 */

/*
 *  Project headers.
 */
#include "network/udp.h"

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
UDP::UDP()
{
}

void
UDP::initialize(const uint16_t& port)
{
    /*
     *  Initialize the class member Arduino Wi-Fi UDP driver object
     *  with the given UDP port.
     */
    udp_.begin(port);
}

std::string
UDP::read(const std::string& ip_remote, const uint16_t& port, const size_t& size)
{
    /*
     *  Read from the given remote IP address and port with the given
     *  read size into a buffer.
     */
    std::shared_ptr<std::vector<char>> buffer = readBuffer(ip_remote, port, size);

    /*
     *  Convert the buffer read into a string.
     */
    if (buffer)
    {
        return std::string(buffer->begin(), buffer->end());
    }
    else
    {
        return "";
    }
}

std::shared_ptr<std::vector<char>>
UDP::readBuffer(const std::string& ip_remote, const uint16_t& port, const size_t& size)
{
    /*
     *  Lock the UDP read mutex.
     */
    std::lock_guard<std::mutex> lock(mutex_udp_read_);

    /*
     *  Declare the read buffer shared pointer as a null pointer.
     */
    std::shared_ptr<std::vector<char>> buffer = nullptr;

    /*
     *  Read into the buffer if the UDP packet remote IP address and port
     *  match the given remote IP address and port.
     */
    if (udp_.parsePacket() && std::string(udp_.remoteIP().toString().c_str()) == ip_remote
            && udp_.remotePort() == port)
    {
        buffer = std::make_shared<std::vector<char>>(size, '0');

        if (buffer)
        {
            udp_.read(buffer->data(), buffer->size());
        }
    }

    /*
     *  Return the buffer shared pointer.
     */
    return buffer;
}

size_t
UDP::write(const std::string& ip_remote, const uint16_t& port, const std::string& data)
{
    /*
     *  Write to the given remote IP address and port with the given string.
     */
    return writeBuffer(ip_remote, port, reinterpret_cast<const uint8_t*>(data.c_str()), data.size());
}

size_t
UDP::writeBuffer(const std::string& ip_remote, const uint16_t& port, const uint8_t* buffer,
        const size_t& size)
{
    /*
     *  Lock the UDP write mutex.
     */
    std::lock_guard<std::mutex> lock(mutex_udp_write_);

    /*
     *  Declare the IP address object.
     */
    IPAddress ip_address;

    /*
     *  Convert the given remote IP address string to the IP address object.
     */
    ip_address.fromString(ip_remote.c_str());

    /*
     *  Begin the UDP packet with the IP address object and the given port.
     */
    udp_.beginPacket(ip_address, port);

    /*
     *  Write the given buffer.
     */
    const size_t bytes = udp_.write(buffer, size);

    /*
     *  End the UDP packet.
     */
    udp_.endPacket();

    /*
     *  Return the bytes written.
     */
    return bytes;
}
}   // namespace firmware
}   // namespace biped
