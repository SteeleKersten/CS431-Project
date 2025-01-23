/*
 * timer.cpp
 *
 *  Created on: Jan 18, 2024
 *      Author: simonyu
 */

/*
 *  External headers.
 */
#include <esp_intr_alloc.h>
#include <soc.h>
#include <timer_group_reg.h>

/*
 *  Project headers.
 */
#include "common/parameter.h"
#include "platform/serial.h"
#include "platform/timer.h"

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
Timer::Timer(const timer_group_t& group, const timer_idx_t& index) : group_(group), index_(index),
        interrupt_handle_(nullptr), interrupt_source_(-1), register_config_(nullptr),
        register_alarm_low_(nullptr), register_alarm_high_(nullptr), register_counter_low_(nullptr),
        register_counter_high_(nullptr), register_interrupt_clear_(nullptr),
        register_interrupt_enable_(nullptr), register_load_low_(nullptr),
        register_load_high_(nullptr)
{
    /*
     *  Initialize the class member timer interrupt clear and enable register pointers by indexing
     *  the corresponding TIMG_..._REG macros in the timer_group_reg header with the class member
     *  timer group, and reinterpret_cast the macros to 32-bit unsigned integer (uint32_t) pointers.
     *
     *  Note that one should always use C++ explicit type casts (static_cast, dynamic_cast, etc.)
     *  when programming in C++ instead of the C-style type cast, as the C++ type casts are checked
     *  by the compiler, whereas C-style casts are not and can fail at runtime.
     *
     *  Learn more about C++ type casting here:
     *  https://cplusplus.com/doc/oldtutorial/typecasting/
     *
     *  Refer to the timer_group_reg header for the timer register macros.
     *
     *  TODO LAB 2 YOUR CODE HERE.
     */

    /*
     *  Switch on timer index.
     */
    switch (index_)
    {
        case TIMER_0:
        {
            /*
             *  Initialize the class member timer configuration, low-bit alarm, high-bit alarm, low-bit
             *  counter, high-bit counter, low-bit load, and high-bit load register pointers by indexing
             *  the corresponding TIMG_..._REG macros in the timer_group_reg header with the class member
             *  timer group, and reinterpret_cast the macros to 32-bit unsigned integer (uint32_t) pointers.
             *
             *  Note that one should always use C++ explicit type casts (static_cast, dynamic_cast, etc.)
             *  when programming in C++ instead of the C-style type cast, as the C++ type casts are checked
             *  by the compiler, whereas C-style casts are not and can fail at runtime.
             *
             *  Note that the macro for the timer low-bit and high-bit counter registers, e.g., for timer 0
             *  in this group, are named TIMG_T0LO_REG and TIMG_T0HI_REG.
             *
             *  Learn more about C++ type casting here:
             *  https://cplusplus.com/doc/oldtutorial/typecasting/
             *
             *  Refer to the timer_group_reg header for the timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            /*
             *  Frame the timer prescaler by first static_cast the timer prescaler parameter in the parameter
             *  header to a 32-bit unsigned integer (uint32_t), then left-shifting the 32-bit prescaler by
             *  TIMG_T..._DIVIDER_S in the timer_group_reg header, and then masking the shifted 32-bit
             *  prescaler with the TIMG_T..._DIVIDER_M macro in the timer_group_reg header.
             *
             *  Note that one should always use C++ explicit type casts (static_cast, dynamic_cast, etc.)
             *  when programming in C++ instead of the C-style type cast, as the C++ type casts are checked
             *  by the compiler, whereas C-style casts are not and can fail at runtime.
             *
             *  Refer to the parameter header for the timer parameters and the timer_group_reg header for the
             *  timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            /*
             *  First, set the timer configuration register to zero.
             *
             *  Using TIMG_T... macros in the timer_group_reg header, set the increase, auto-reload, level
             *  interrupt enable, and alarm enable bits in the timer configuration register. Additionally, also
             *  set the prescaler bits framed above to the timer configuration register.
             *
             *  Why should we use level-triggered interrupts for the hardware timer? What happens if
             *  the system miss observing the rising of the timer alarm?
             *
             *  Learn more about level-triggered vs. edge-triggered interrupts here:
             *  https://www.garystringham.com/level-triggered-vs-edge-triggered-interrupts/
             *
             *  Remember to dereference the register pointers to set the value of the registers.
             *
             *  Refer to the timer_group_reg header for the timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            /*
             *  Switch on timer group.
             */
            switch (group_)
            {
                case TIMER_GROUP_0:
                {
                    /*
                     *  Set the class member interrupt source to the corresponding ETS_TG..._LEVEL_INTR_SOURCE
                     *  macros in the soc header.
                     *
                     *  Why should we use level-triggered interrupts for the hardware timer? What happens if
                     *  the system miss observing the rising of the timer alarm?
                     *
                     *  Learn more about level-triggered vs. edge-triggered interrupts here:
                     *  https://www.garystringham.com/level-triggered-vs-edge-triggered-interrupts/
                     *
                     *  Refer to the soc header for the interrupt source macros.
                     *
                     *  TODO LAB 2 YOUR CODE HERE.
                     */

                    break;
                }
                case TIMER_GROUP_1:
                {
                    /*
                     *  Set the class member interrupt source to the corresponding ETS_TG..._LEVEL_INTR_SOURCE
                     *  macros in the soc header.
                     *
                     *  Why should we use level-triggered interrupts for the hardware timer? What happens if
                     *  the system miss observing the rising of the timer alarm?
                     *
                     *  Learn more about level-triggered vs. edge-triggered interrupts here:
                     *  https://www.garystringham.com/level-triggered-vs-edge-triggered-interrupts/
                     *
                     *  Refer to the soc header for the interrupt source macros.
                     *
                     *  TODO LAB 2 YOUR CODE HERE.
                     */

                    break;
                }
                default:
                {
                    break;
                }
            }

            break;
        }
        case TIMER_1:
        {
            /*
             *  Initialize the class member timer configuration, low-bit alarm, high-bit alarm, low-bit
             *  counter, high-bit counter, low-bit load, and high-bit load register pointers by indexing
             *  the corresponding TIMG_..._REG macros in the timer_group_reg header with the class member
             *  timer group, and reinterpret_cast the macros to 32-bit unsigned integer (uint32_t) pointers.
             *
             *  Note that one should always use C++ explicit type casts (static_cast, dynamic_cast, etc.)
             *  when programming in C++ instead of the C-style type cast, as the C++ type casts are checked
             *  by the compiler, whereas C-style casts are not and can fail at runtime.
             *
             *  Note that the macro for the timer low-bit and high-bit counter registers, e.g., for timer 0
             *  in this group, are named TIMG_T0LO_REG and TIMG_T0HI_REG.
             *
             *  Learn more about C++ type casting here:
             *  https://cplusplus.com/doc/oldtutorial/typecasting/
             *
             *  Refer to the timer_group_reg header for the timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            /*
             *  Frame the timer prescaler by first static_cast the timer prescaler parameter in the parameter
             *  header to a 32-bit unsigned integer (uint32_t), then left-shifting the 32-bit prescaler by
             *  TIMG_T..._DIVIDER_S in the timer_group_reg header, and then masking the shifted 32-bit
             *  prescaler with the TIMG_T..._DIVIDER_M macro in the timer_group_reg header.
             *
             *  Note that one should always use C++ explicit type casts (static_cast, dynamic_cast, etc.)
             *  when programming in C++ instead of the C-style type cast, as the C++ type casts are checked
             *  by the compiler, whereas C-style casts are not and can fail at runtime.
             *
             *  Refer to the parameter header for the timer parameters and the timer_group_reg header for the
             *  timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            /*
             *  First, set the timer configuration register to zero.
             *
             *  Using TIMG_T... macros in the timer_group_reg header, set the increase, auto-reload, level
             *  interrupt enable, and alarm enable bits in the timer configuration register. Additionally, also
             *  set the prescaler bits framed above to the timer configuration register.
             *
             *  Why should we use level-triggered interrupts for the hardware timer? What happens if
             *  the system miss observing the rising of the timer alarm?
             *
             *  Learn more about level-triggered vs. edge-triggered interrupts here:
             *  https://www.garystringham.com/level-triggered-vs-edge-triggered-interrupts/
             *
             *  Remember to dereference the register pointers to set the value of the registers.
             *
             *  Refer to the timer_group_reg header for the timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            /*
             *  Switch on timer group.
             */
            switch (group_)
            {
                case TIMER_GROUP_0:
                {
                    /*
                     *  Set the class member interrupt source to the corresponding ETS_TG..._LEVEL_INTR_SOURCE
                     *  macros in the soc header.
                     *
                     *  Why should we use level-triggered interrupts for the hardware timer? What happens if
                     *  the system miss observing the rising of the timer alarm?
                     *
                     *  Learn more about level-triggered vs. edge-triggered interrupts here:
                     *  https://www.garystringham.com/level-triggered-vs-edge-triggered-interrupts/
                     *
                     *  Refer to the soc header for the interrupt source macros.
                     *
                     *  TODO LAB 2 YOUR CODE HERE.
                     */

                    break;
                }
                case TIMER_GROUP_1:
                {
                    /*
                     *  Set the class member interrupt source to the corresponding ETS_TG..._LEVEL_INTR_SOURCE
                     *  macros in the soc header.
                     *
                     *  Why should we use level-triggered interrupts for the hardware timer? What happens if
                     *  the system miss observing the rising of the timer alarm?
                     *
                     *  Learn more about level-triggered vs. edge-triggered interrupts here:
                     *  https://www.garystringham.com/level-triggered-vs-edge-triggered-interrupts/
                     *
                     *  Refer to the soc header for the interrupt source macros.
                     *
                     *  TODO LAB 2 YOUR CODE HERE.
                     */

                    break;
                }
                default:
                {
                    break;
                }
            }

            break;
        }
        default:
        {
            /*
             *  Print an error message to serial for invalid timer index.
             */
            Serial(LogLevel::error) << "Invalid timer index.";
            return;
        }
    }

    /*
     *  Set the timer low-bit counter, high-bit counter, low-bit load, and high-bit load
     *  registers to zeros.
     *
     *  Remember to dereference the register pointers to set the value of the registers.
     *
     *  TODO LAB 2 YOUR CODE HERE.
     */
}

bool
Timer::attachInterrupt(void
(*handler)(void*), void* arg)
{
    /*
     *  If the class member interrupt source is negative, return false.
     *
     *  TODO LAB 2 YOUR CODE HERE.
     */

    /*
     *  Using the ESP-IDF esp_intr_alloc function in the esp_intr_alloc header, allocate
     *  the timer interrupt handler using the class member interrupt source, the given
     *  interrupt handler, the given interrupt handler function argument pointer, and a
     *  pointer to the class member interrupt handle pointer. The interrupt flags should
     *  be zero. If the return value of the esp_intr_alloc function is not the ESP_OK
     *  macro, return false.
     *
     *  Note that the last argument of the esp_intr_alloc function is the pointer to the
     *  pointer, not the pointer itself.
     *
     *  Refer to the esp_intr_alloc header for the ESP-IDF interrupt functions.
     *
     *  TODO LAB 2 YOUR CODE HERE.
     */

    /*
     *  Using the ESP-IDF esp_intr_enable function in the esp_intr_alloc header, enable
     *  the allocated timer interrupt handler using the class member interrupt handle
     *  pointer. If the return value of the esp_intr_enable function is not the ESP_OK
     *  macro, return false.
     *
     *  Refer to the esp_intr_alloc header for the ESP-IDF interrupt functions.
     *
     *  TODO LAB 2 YOUR CODE HERE.
     */

    /*
     *  Switch on timer index.
     */
    switch (index_)
    {
        case TIMER_0:
        {
            /*
             *  Without disturbing other existing bits in the timer interrupt enable
             *  register, using the TIMG_T... macros in the timer_group_reg header,
             *  set the interrupt enable bit in the timer interrupt enable register,
             *  thus enabling the interrupt on the timer.
             *
             *  Refer to the timer_group_reg header for the timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            break;
        }
        case TIMER_1:
        {
            /*
             *  Without disturbing other existing bits in the timer interrupt enable
             *  register, using the TIMG_T... macros in the timer_group_reg header,
             *  set the interrupt enable bit in the timer interrupt enable register,
             *  thus enabling the interrupt on the timer.
             *
             *  Refer to the timer_group_reg header for the timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            break;
        }
        default:
        {
            break;
        }
    }

    return true;
}

void IRAM_ATTR
Timer::clearInterrupt()
{
    switch (index_)
    {
        case TIMER_0:
        {
            /*
             *  Without disturbing other existing bits in the timer interrupt clear
             *  and configuration registers, using the TIMG_T... macros in the timer_group_reg
             *  header, set the interrupt clear bit in the timer interrupt clear register,
             *  thus clearing the timer interrupt, and set the alarm enable bit in the
             *  timer configuration register, thus re-enabling the timer alarm.
             *
             *  Remember that we already enabled the timer alarm in the constructor. However,
             *  the timer alarm gets disabled every time the timer alarm goes off. Therefore,
             *  we need to re-enable the timer alarm in the timer interrupt handler, which
             *  calls this function.
             *
             *  Refer to the timer_group_reg header for the timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            break;
        }
        case TIMER_1:
        {
            /*
             *  Without disturbing other existing bits in the timer interrupt clear
             *  and configuration registers, using the TIMG_T... macros in the timer_group_reg
             *  header, set the interrupt clear bit in the timer interrupt clear register,
             *  thus clearing the timer interrupt, and set the alarm enable bit in the
             *  timer configuration register, thus re-enabling the timer alarm.
             *
             *  Remember that we already enabled the timer alarm in the constructor. However,
             *  the timer alarm gets disabled every time the timer alarm goes off. Therefore,
             *  we need to re-enable the timer alarm in the timer interrupt handler, which
             *  calls this function.
             *
             *  Refer to the timer_group_reg header for the timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            break;
        }
        default:
        {
            break;
        }
    }
}

bool
Timer::detachInterrupt()
{
    switch (index_)
    {
        case TIMER_0:
        {
            /*
             *  Without disturbing other existing bits in the timer interrupt enable
             *  register, using the TIMG_T... macros in the timer_group_reg header,
             *  unset the interrupt enable bit in the timer interrupt enable register,
             *  thus disabling the interrupt on the timer.
             *
             *  Refer to the timer_group_reg header for the timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            break;
        }
        case TIMER_1:
        {
            /*
             *  Without disturbing other existing bits in the timer interrupt enable
             *  register, using the TIMG_T... macros in the timer_group_reg header,
             *  unset the interrupt enable bit in the timer interrupt enable register,
             *  thus disabling the interrupt on the timer.
             *
             *  Refer to the timer_group_reg header for the timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            break;
        }
        default:
        {
            break;
        }
    }

    /*
     *  Using the ESP-IDF esp_intr_disable function in the esp_intr_alloc header,
     *  disabled the allocated timer interrupt handler using the class member interrupt
     *  handle pointer. If the return value of the esp_intr_disable function is not the
     *  ESP_OK macro, return false.
     *
     *  Refer to the esp_intr_alloc header for the ESP-IDF interrupt functions.
     *
     *  TODO LAB 2 YOUR CODE HERE.
     */

    /*
     *  Using the ESP-IDF esp_intr_free function in the esp_intr_alloc header, free the
     *  allocated timer interrupt handler using the class member interrupt handle pointer.
     *  If the return value of the esp_intr_free function is not the ESP_OK macro, return
     *  false.
     *
     *  Refer to the esp_intr_alloc header for the ESP-IDF interrupt functions.
     *
     *  TODO LAB 2 YOUR CODE HERE.
     */

    /*
     *  Set the class member interrupt handle pointer to a null pointer.
     *
     *  Note that due to the existence of smart pointers in C++ (unique pointer,
     *  shared pointer, etc.) One should always use the nullptr keyword to denote
     *  all null pointer values when programming in C++, instead of the C macro NULL.
     *
     *  TODO LAB 2 YOUR CODE HERE.
     */

    /*
     *  Return true.
     */
    return true;
}

void
Timer::disable()
{
    switch (index_)
    {
        case TIMER_0:
        {
            /*
             *  Without disturbing other existing bits in the timer configuration register,
             *  using the TIMG_T... macros in the timer_group_reg header, unset the enable
             *  bit in the timer configuration register, thus disabling the timer.
             *
             *  Refer to the timer_group_reg header for the timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            break;
        }
        case TIMER_1:
        {
            /*
             *  Without disturbing other existing bits in the timer configuration register,
             *  using the TIMG_T... macros in the timer_group_reg header, unset the enable
             *  bit in the timer configuration register, thus disabling the timer.
             *
             *  Refer to the timer_group_reg header for the timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            break;
        }
        default:
        {
            break;
        }
    }
}

void
Timer::enable()
{
    switch (index_)
    {
        case TIMER_0:
        {
            /*
             *  Without disturbing other existing bits in the timer configuration register,
             *  using the TIMG_T... macros in the timer_group_reg header, set the enable
             *  bit in the timer configuration register, thus enabling the timer.
             *
             *  Refer to the timer_group_reg header for the timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            break;
        }
        case TIMER_1:
        {
            /*
             *  Without disturbing other existing bits in the timer configuration register,
             *  using the TIMG_T... macros in the timer_group_reg header, set the enable
             *  bit in the timer configuration register, thus enabling the timer.
             *
             *  Refer to the timer_group_reg header for the timer register macros.
             *
             *  TODO LAB 2 YOUR CODE HERE.
             */

            break;
        }
        default:
        {
            break;
        }
    }
}

void
Timer::setInterval(const uint64_t& interval)
{
    /*
     *  Split the given 64-bit timer interval into two 32-bit unsigned integers
     *  (uint32_t). Set the lower 32-bit values in the given timer interval to the timer
     *  low-bit alarm register and the higher 32-bit values in the given timer interval
     *  to the timer high-bit alarm register.
     *
     *  Remember to dereference the register pointers to set the value of the registers.
     *
     *  TODO LAB 2 YOUR CODE HERE.
     */
}
}   // namespace firmware
}   // namespace biped
