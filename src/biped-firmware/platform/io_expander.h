/**
 *  @file   io_expander.h
 *  @author Simon Yu
 *  @date   12/06/2022
 *  @brief  I/O expander class header.
 *
 *  This file defines the I/O expander class.
 */

/*
 *  Include guard.
 */
#ifndef PLATFORM_IO_EXPANDER_H_
#define PLATFORM_IO_EXPANDER_H_

/*
 *  External headers.
 */
#include <MCP23018.h>
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
 *  @brief  I/O expander class.
 *
 *  This class provides functions for performing digital
 *  read and write operations on the I/O expander pins,
 *  setting I/O expander pin modes, hosting an interrupt
 *  service for the I/O expander pins, as well as for
 *  attaching and detaching interrupt handlers to and from
 *  the I/O expander pins.
 */
class IOExpander
{
public:

    /**
     *  @param  address I2C address.
     *  @brief  I/O expander class constructor.
     *
     *  This constructor initializes all class member variables and
     *  the Arduino I/O expander driver.
     */
    IOExpander(const uint8_t& address);

    /**
     *  @return Arduino I/O expander driver shared pointer.
     *  @brief  Get Arduino I/O expander driver shared pointer.
     *
     *  This function returns the Arduino I/O expander driver shared
     *  pointer.
     */
    std::shared_ptr<MCP23018>
    get() const;

    /**
     *  @return Arduino I/O expander driver raw pointer.
     *  @brief  Get Arduino I/O expander driver raw pointer.
     *
     *  This function returns the Arduino I/O expander driver raw
     *  pointer.
     */
    MCP23018*
    getRaw() const;

    /**
     *  @param  pin I/O expander pin.
     *  @param  handler Interrupt handler function pointer.
     *  @param  arg Interrupt handler function argument pointer.
     *  @param  mode Interrupt mode.
     *  @brief  Attach an interrupt handler to the given port A pin.
     *
     *  This function attaches the given interrupt handler to the given
     *  port A pin with the given interrupt mode. The given function
     *  argument pointer is passed to the given interrupt handler when
     *  it is called.
     *
     *  MCP23018 is a two-port I/O expander with port A (GPA) pins and
     *  port B (GPB) pins.
     *
     *  Learn more about the MCP23018 I/O expander here:
     *  https://ww1.microchip.com/downloads/en/devicedoc/22103a.pdf
     */
    void
    attachInterruptPortA(const uint8_t& pin, void
    (*handler)(void*), void* arg, const int& mode);

    /**
     *  @param  pin I/O expander pin.
     *  @param  handler Interrupt handler function pointer.
     *  @param  arg Interrupt handler function argument pointer.
     *  @param  mode Interrupt mode.
     *  @brief  Attach an interrupt handler to the given port B pin.
     *
     *  This function attaches the given interrupt handler to the given
     *  port B pin with the given interrupt mode. The given function
     *  argument pointer is passed to the given interrupt handler when
     *  it is called.
     *
     *  MCP23018 is a two-port I/O expander with port A (GPA) pins and
     *  port B (GPB) pins.
     *
     *  Learn more about the MCP23018 I/O expander here:
     *  https://ww1.microchip.com/downloads/en/devicedoc/22103a.pdf
     */
    void
    attachInterruptPortB(const uint8_t& pin, void
    (*handler)(void*), void* arg, const int& mode);

    /**
     *  @param  pin I/O expander pin.
     *  @brief  Detach all interrupt handlers from the given port A pin.
     *
     *  This function detaches all interrupt handlers from the given port
     *  A pin.
     *
     *  MCP23018 is a two-port I/O expander with port A (GPA) pins and
     *  port B (GPB) pins.
     *
     *  Learn more about the MCP23018 I/O expander here:
     *  https://ww1.microchip.com/downloads/en/devicedoc/22103a.pdf
     */
    void
    detachInterruptPortA(const uint8_t& pin);

    /**
     *  @param  pin I/O expander pin.
     *  @brief  Detach all interrupt handlers from the given port B pin.
     *
     *  This function detaches all interrupt handlers from the given port
     *  B pin.
     *
     *  MCP23018 is a two-port I/O expander with port A (GPA) pins and
     *  port B (GPB) pins.
     *
     *  Learn more about the MCP23018 I/O expander here:
     *  https://ww1.microchip.com/downloads/en/devicedoc/22103a.pdf
     */
    void
    detachInterruptPortB(const uint8_t& pin);

    /**
     *  @param  pin I/O expander pin.
     *  @param  mode Pin mode.
     *  @brief  Set pin mode of the given port A pin.
     *
     *  This function sets the pin mode of the given port A pin.
     *
     *  MCP23018 is a two-port I/O expander with port A (GPA) pins and
     *  port B (GPB) pins.
     *
     *  Learn more about the MCP23018 I/O expander here:
     *  https://ww1.microchip.com/downloads/en/devicedoc/22103a.pdf
     */
    void
    pinModePortA(const uint8_t& pin, const uint8_t& mode);

    /**
     *  @param  pin I/O expander pin.
     *  @param  mode Pin mode.
     *  @brief  Set pin mode of the given port B pin.
     *
     *  This function sets the pin mode of the given port B pin.
     *
     *  MCP23018 is a two-port I/O expander with port A (GPA) pins and
     *  port B (GPB) pins.
     *
     *  Learn more about the MCP23018 I/O expander here:
     *  https://ww1.microchip.com/downloads/en/devicedoc/22103a.pdf
     */
    void
    pinModePortB(const uint8_t& pin, const uint8_t& mode);

    /**
     *  @param  pin I/O expander pin.
     *  @return Pin state.
     *  @brief  Perform a digital read on the given port A pin.
     *
     *  This function performs a digital read on the given port
     *  A pin and returns the state of the pin read.
     *
     *  MCP23018 is a two-port I/O expander with port A (GPA) pins and
     *  port B (GPB) pins. Since MCP23018 is also a digital I/O expander,
     *  analog operations are not available.
     *
     *  Learn more about the MCP23018 I/O expander here:
     *  https://ww1.microchip.com/downloads/en/devicedoc/22103a.pdf
     */
    bool
    digitalReadPortA(const uint8_t& pin);

    /**
     *  @param  pin I/O expander pin.
     *  @return Pin state.
     *  @brief  Perform a digital read on the given port B pin.
     *
     *  This function performs a digital read on the given port
     *  B pin and returns the state of the pin read.
     *
     *  MCP23018 is a two-port I/O expander with port A (GPA) pins and
     *  port B (GPB) pins. Since MCP23018 is also a digital I/O expander,
     *  analog operations are not available.
     *
     *  Learn more about the MCP23018 I/O expander here:
     *  https://ww1.microchip.com/downloads/en/devicedoc/22103a.pdf
     */
    bool
    digitalReadPortB(const uint8_t& pin);

    /**
     *  @param  pin I/O expander pin.
     *  @param  state Pin state.
     *  @brief  Perform a digital write on the given port A pin.
     *
     *  This function performs a digital write on the given port A pin with
     *  the given pin state.
     *
     *  MCP23018 is a two-port I/O expander with port A (GPA) pins and
     *  port B (GPB) pins. Since MCP23018 is also a digital I/O expander,
     *  analog operations are not available.
     *
     *  Learn more about the MCP23018 I/O expander here:
     *  https://ww1.microchip.com/downloads/en/devicedoc/22103a.pdf
     */
    void
    digitalWritePortA(const uint8_t& pin, const bool& state);

    /**
     *  @param  pin I/O expander pin.
     *  @param  state Pin state.
     *  @brief  Perform a digital write on the given port B pin.
     *
     *  This function performs a digital write on the given port B pin with
     *  the given pin state.
     *
     *  MCP23018 is a two-port I/O expander with port A (GPA) pins and
     *  port B (GPB) pins. Since MCP23018 is also a digital I/O expander,
     *  analog operations are not available.
     *
     *  Learn more about the MCP23018 I/O expander here:
     *  https://ww1.microchip.com/downloads/en/devicedoc/22103a.pdf
     */
    void
    digitalWritePortB(const uint8_t& pin, const bool& state);

    /**
     *  @brief  I/O expander interrupt service callback function.
     *
     *  This function hosts the I/O expander interrupt service. The function
     *  reads the interrupt flags and captures from the I/O expander registers
     *  and services the interrupts based on the interrupt flags and captures.
     *
     *  Learn more about the MCP23018 I/O expander here:
     *  https://ww1.microchip.com/downloads/en/devicedoc/22103a.pdf
     */
    void IRAM_ATTR
    onInterrupt();

private:

    /**
     *  @brief  Interrupt handler struct.
     *
     *  This struct contains interrupt handler entries, such as the interrupt
     *  handler function pointer, interrupt handler function argument pointer,
     *  and etc.
     */
    struct InterruptHandler
    {
        void
        (*handler)(void*);  //!< Interrupt handler function pointer.
        void *arg;  //!< Interrupt handler function argument pointer.
        int mode;   //!< Interrupt mode.

        /**
         *  @brief  Interrupt handler struct constructor.
         *
         *  This constructor initializes all interrupt handler struct entries.
         */
        InterruptHandler() : handler(nullptr), arg(nullptr), mode(DISABLED)
        {
        }
    };

    /**
     *  @param  flags I/O expander interrupt flags.
     *  @param  captures I/O expander interrupt captures.
     *  @param  interrupt_handlers I/O expander interrupt handler vector.
     *  @brief  Handle I/O expander interrupts.
     *
     *  This function services the I/O expander interrupts by examining the
     *  given I/O expander interrupt flags and captures, and subsequently
     *  calling the corresponding interrupt handlers from the given I/O
     *  expander interrupt handler vector.
     *
     *  Learn more about the MCP23018 I/O expander here:
     *  https://ww1.microchip.com/downloads/en/devicedoc/22103a.pdf
     */
    void
    serviceInterrupt(const uint8_t& flags, const uint8_t& captures,
            const std::vector<InterruptHandler>& interrupt_handlers);

    std::vector<InterruptHandler> interrupt_handlers_port_a_;  //!< Port A interrupt handler vector.
    std::vector<InterruptHandler> interrupt_handlers_port_b_;  //!< Port B interrupt handler vector.
    std::shared_ptr<MCP23018> mcp23018_; //!< Arduino MCP23018 I/O expander driver shared pointer.
};
}   // namespace firmware
}   // namespace biped

#endif  // PLATFORM_IO_EXPANDER_H_
