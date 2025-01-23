/**
 *  @file   timer.h
 *  @author Simon Yu
 *  @date   01/18/2024
 *  @brief  Timer class header.
 *
 *  This file defines the timer class.
 */

/*
 *  Include guard.
 */
#ifndef PLATFORM_TIMER_H_
#define PLATFORM_TIMER_H_

/*
 *  External headers.
 */
#include <driver/timer.h>

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
 *  @brief  Timer class.
 *
 *  This class provides functions for initializing
 *  hardware timers on the ESP32 SoC, setting timer
 *  interval, and attaching timer interrupt handlers
 *  to the timers.
 */
class Timer
{
public:

    /**
     *  @param  group Timer group.
     *  @param  index Timer index in the timer group.
     *  @brief  Timer class constructor.
     *
     *  This constructor initializes all class member variables.
     *
     *  The ESP32 SoC has four hardware timers. The hardware timers
     *  are divided into two timer groups, each of which has two
     *  hardware timers.
     *
     *  Learn more about the ESP32 hardware timers in Lecture 4 and here:
     *  https://circuitdigest.com/microcontroller-projects/esp32-timers-and-timer-interrupts
     */
    Timer(const timer_group_t& group, const timer_idx_t& index);

    /**
     *  @param  handler Interrupt handler function pointer.
     *  @param  arg Interrupt handler function argument pointer, default to a null pointer.
     *  @return Whether the given interrupt handler was attached successfully.
     *  @brief  Attach an interrupt handler to the hardware timer.
     *
     *  This function attaches the given interrupt handler to the hardware
     *  timer. The given function argument pointer is passed to the
     *  given interrupt handler when it is called.
     */
    bool
    attachInterrupt(void
    (*handler)(void*), void* arg = nullptr);

    /**
     *  @brief  Clear the timer interrupt.
     *
     *  This function clears the generated timer interrupt.
     */
    void IRAM_ATTR
    clearInterrupt();

    /**
     *  @return Whether the interrupt handler was detached successfully.
     *  @brief  Detach the interrupt handler from the hardware timer.
     *
     *  This function detaches the interrupt handler from the hardware
     *  timer.
     */
    bool
    detachInterrupt();

    /**
     *  @brief  Disable the hardware timer.
     *
     *  This function disables the hardware timer.
     */
    void
    disable();

    /**
     *  @brief  Enable the hardware timer.
     *
     *  This function enables the hardware timer.
     */
    void
    enable();

    /**
     *  @param  interval hardware timer interval, in microseconds.
     *  @brief  Set the hardware timer interval.
     *
     *  This function sets the hardware timer interval, i.e., the
     *  period at which the hardware timer alarm is triggered.
     */
    void
    setInterval(const uint64_t& interval);

private:

    timer_group_t group_;   //!< Timer group.
    timer_idx_t index_; //!< Timer index in the timer group.
    intr_handle_t interrupt_handle_;    //!< Interrupt handle pointer.
    int interrupt_source_;  //!< Interrupt source.
    volatile uint32_t *register_config_;    //!< Timer configuration register pointer.
    volatile uint32_t *register_alarm_low_; //!< Timer low-bit alarm register pointer.
    volatile uint32_t *register_alarm_high_;    //!< Timer high-bit alarm register pointer.
    volatile uint32_t *register_counter_low_;   //!< Timer low-bit counter register pointer.
    volatile uint32_t *register_counter_high_;  //!< Timer high-bit counter register pointer.
    volatile uint32_t *register_interrupt_clear_;   //!< Timer interrupt clear register pointer.
    volatile uint32_t *register_interrupt_enable_;  //!< Timer interrupt enable register pointer.
    volatile uint32_t *register_load_low_;  //!< Timer low-bit load register pointer.
    volatile uint32_t *register_load_high_; //!< Timer high-bit load register pointer.
};
}   // namespace firmware
}   // namespace biped

#endif /* PLATFORM_TIMER_H_ */
