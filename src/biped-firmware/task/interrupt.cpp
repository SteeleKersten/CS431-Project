/**
 *  @file   interrupt.cpp
 *  @author Simon Yu
 *  @date   12/03/2022
 *  @brief  Interrupt function source.
 *
 *  This file implements the interrupt functions.
 */

/*
 *  External headers.
 */
#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <gpio_struct.h>
#include <hal/misc.h>

/*
 *  Project headers.
 */
#include "common/global.h"
#include "common/parameter.h"
#include "common/pin.h"
#include "planner/planner.h"
#include "platform/serial.h"
#include "platform/timer.h"
#include "sensor/sensor.h"
#include "task/interrupt.h"

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
void
attachInterrupt(const uint8_t& pin, void
(*handler)(void), const int& mode)
{
    /*
     *  Using the ESP-IDF gpio_isr_handler_add function, attach the given
     *  interrupt handler to the given ESP32 pin.
     */
    esp_err_t result = gpio_isr_handler_add(static_cast<gpio_num_t>(pin),
            reinterpret_cast<gpio_isr_t>(handler), nullptr);

    /*
     *  Print an error message to serial and return upon failure.
     */
    if (result != ESP_OK)
    {
        Serial(LogLevel::error) << "Failed to add interrupt handler.";
        return;
    }

    /*
     *  Set interrupt type based on the given interrupt mode.
     */
    switch (mode)
    {
        case RISING:
        {
            result = gpio_set_intr_type(static_cast<gpio_num_t>(pin), GPIO_INTR_POSEDGE);
            break;
        }
        case FALLING:
        {
            result = gpio_set_intr_type(static_cast<gpio_num_t>(pin), GPIO_INTR_NEGEDGE);
            break;
        }
        case CHANGE:
        {
            result = gpio_set_intr_type(static_cast<gpio_num_t>(pin), GPIO_INTR_ANYEDGE);
            break;
        }
        case ONLOW:
        {
            result = gpio_set_intr_type(static_cast<gpio_num_t>(pin), GPIO_INTR_LOW_LEVEL);
            break;
        }
        case ONHIGH:
        {
            result = gpio_set_intr_type(static_cast<gpio_num_t>(pin), GPIO_INTR_HIGH_LEVEL);
            break;
        }
        case ONLOW_WE:
        {
            result = gpio_set_intr_type(static_cast<gpio_num_t>(pin), GPIO_INTR_LOW_LEVEL);
            break;
        }
        case ONHIGH_WE:
        {
            result = gpio_set_intr_type(static_cast<gpio_num_t>(pin), GPIO_INTR_HIGH_LEVEL);
            break;
        }
        default:
        {
            /*
             *  Print an error message to serial for unknown interrupt mode.
             */
            Serial(LogLevel::error) << "Unknown interrupt mode: " << mode << ".";
            break;
        }
    }

    /*
     *  Print an error message to serial and return upon failure.
     */
    if (result != ESP_OK)
    {
        Serial(LogLevel::error) << "Failed to set interrupt mode.";
    }
}

int IRAM_ATTR
digitalReadFromISR(uint8_t pin)
{
    /*
     *  static_cast the given ESP32 pin into a 32-bit unsigned integer.
     */
    uint32_t pin_gpio = static_cast<uint32_t>(pin);

    /*
     *  Read the given ESP32 pin and return the pin state read.
     */
    if (pin_gpio < 32)
    {
        return (GPIO.in >> pin_gpio) & 0x1;
    }
    else
    {
        return (HAL_FORCE_READ_U32_REG_FIELD(GPIO.in1, data) >> (pin_gpio - 32)) & 0x1;
    }
}

void IRAM_ATTR
encoderLeftAInterruptHandler()
{
    /*
     *  If the sensor global shared pointer is not a null pointer, using
     *  the sensor global shared pointer, call the left encoder A callback
     *  function.
     *
     *  Refer to the sensor header for the sensor functions.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
}

void IRAM_ATTR
encoderLeftBInterruptHandler()
{
    /*
     *  If the sensor global shared pointer is not a null pointer, using
     *  the sensor global shared pointer, call the left encoder B callback
     *  function.
     *
     *  Refer to the sensor header for the sensor functions.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
}

void IRAM_ATTR
encoderRightAInterruptHandler()
{
    /*
     *  If the sensor global shared pointer is not a null pointer, using
     *  the sensor global shared pointer, call the right encoder A callback
     *  function.
     *
     *  Refer to the sensor header for the sensor functions.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
}

void IRAM_ATTR
encoderRightBInterruptHandler()
{
    /*
     *  If the sensor global shared pointer is not a null pointer, using
     *  the sensor global shared pointer, call the right encoder B callback
     *  function.
     *
     *  Refer to the sensor header for the sensor functions.
     *
     *  TODO LAB 6 YOUR CODE HERE.
     */
}

void IRAM_ATTR
ioExpanderAInterruptHandler()
{
    /*
     *  Using the Arduino detachInterrupt function, detach this very interrupt
     *  handler.
     *
     *  Why? Remember that this interrupt handler has been attached in on-high
     *  interrupt mode in order to handle level-triggered interrupts. Image a push
     *  button connected to one of the I/O expander pins. An interrupt is generated
     *  every time the push button is pressed. What would happen if the push button
     *  is held down? Can the I/O expander interrupt tasks woken below still run?
     *
     *  Learn more about level-triggered vs. edge-triggered interrupts here:
     *  https://www.garystringham.com/level-triggered-vs-edge-triggered-interrupts/
     *
     *  Additionally, always wrap the ESP32 interrupt pins around the
     *  digitalPinToInterrupt macro to ensure that the given pin numbers are within
     *  the ESP32 interrupt pin range.
     *
     *  Refer to the pin header for the ESP32 pins.
     *
     *  TODO LAB 4 YOUR CODE HERE.
     */

    /*
     *  If the I/O expander A interrupt service task handle global pointer is not
     *  a null pointer, using the FreeRTOS vTaskNotifyGiveFromISR function, wake
     *  the I/O expander A interrupt service task. The pxHigherPriorityTaskWoken
     *  should be set to a null pointer.
     *
     *  Note that due to the existence of smart pointers in C++ (unique pointer,
     *  shared pointer, etc.) One should always use the nullptr keyword to denote
     *  all null pointer values when programming in C++, instead of the C macro NULL.
     *
     *  Refer to the global header for the global variables and the FreeRTOS task
     *  header for the FreeRTOS task functions.
     *
     *  TODO LAB 3 YOUR CODE HERE.
     */
}

void IRAM_ATTR
ioExpanderBInterruptHandler()
{
    /*
     *  Using the Arduino detachInterrupt function, detach this very interrupt
     *  handler.
     *
     *  Why? Remember that this interrupt handler has been attached in on-high
     *  interrupt mode in order to handle level-triggered interrupts. Image a push
     *  button connected to one of the I/O expander pins. An interrupt is generated
     *  every time the push button is pressed. What would happen if the push button
     *  is held down? Can the I/O expander interrupt tasks woken below still run?
     *
     *  Learn more about level-triggered vs. edge-triggered interrupts here:
     *  https://www.garystringham.com/level-triggered-vs-edge-triggered-interrupts/
     *
     *  Additionally, always wrap the ESP32 interrupt pins around the
     *  digitalPinToInterrupt macro to ensure that the given pin numbers are within
     *  the ESP32 interrupt pin range.
     *
     *  Refer to the pin header for the ESP32 pins.
     *
     *  TODO LAB 4 YOUR CODE HERE.
     */

    /*
     *  If the I/O expander B interrupt service task handle global pointer is not
     *  a null pointer, using the FreeRTOS vTaskNotifyGiveFromISR function, wake
     *  the I/O expander B interrupt service task. The pxHigherPriorityTaskWoken
     *  should be set to a null pointer.
     *
     *  Note that due to the existence of smart pointers in C++ (unique pointer,
     *  shared pointer, etc.) One should always use the nullptr keyword to denote
     *  all null pointer values when programming in C++, instead of the C macro NULL.
     *
     *  Refer to the global header for the global variables and the FreeRTOS task
     *  header for the FreeRTOS task functions.
     *
     *  TODO LAB 3 YOUR CODE HERE.
     */
}

void IRAM_ATTR
pushButtonAInterruptHandler(void* arg)
{
    /*
     *  If the planner global shared pointer is not a null pointer, using the planner
     *  global shared pointer, start the planner.
     *
     *  Refer to the planner header for the pure virtual planner functions.
     *
     *  TODO LAB 8 YOUR CODE HERE.
     */
}

void IRAM_ATTR
pushButtonBInterruptHandler(void* arg)
{
}

void IRAM_ATTR
pushButtonCInterruptHandler(void* arg)
{
}

void IRAM_ATTR
timerInterruptHandler(void* arg)
{
    /*
     *  Declare static start time point and set to 0.
     */
    static unsigned long time_point_start = 0;

    /*
     *  Calculate the real-time task interval by subtracting the current time in
     *  microseconds, obtained using the Arduino micros timing function, with the
     *  start time point static local variable and storing the result to the real-time
     *  task interval global variable.
     *
     *  Then, update the static start time point local variable by setting it to the
     *  current time in microseconds, obtained using the Arduino micros timing function.
     *
     *  Refer to the global header for the global variables.
     *
     *  TODO LAB 2 YOUR CODE HERE.
     */

    /*
     *  If the real-time task handle global pointer is not a null pointer, wake
     *  the real-time task using the FreeRTOS vTaskNotifyGiveFromISR function.
     *  The pxHigherPriorityTaskWoken should be set to a null pointer.
     *
     *  Note that due to the existence of smart pointers in C++ (unique pointer,
     *  shared pointer, etc.) One should always use the nullptr keyword to denote
     *  all null pointer values when programming in C++, instead of the C macro NULL.
     *
     *  Refer to the global header for the global variables and the FreeRTOS task
     *  header for the FreeRTOS task functions.
     *
     *  TODO LAB 3 YOUR CODE HERE.
     */

    /*
     *  If the timer global shared pointer is not a null pointer,
     *  using the timer global shared pointer, clear the timer
     *  interrupt.
     *
     *  TODO LAB 2 YOUR CODE HERE.
     */
}
}   // namespace firmware
}   // namespace biped
