/**
 *  @file   task.cpp
 *  @author Simon Yu
 *  @date   12/03/2022
 *  @brief  Task function source.
 *
 *  This file implements the task functions.
 */

/*
 *  External headers.
 */
#include <Esp.h>
#include <esp32-hal-gpio.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/portmacro.h>
#include <freertos/projdefs.h>
#include <vector>
#include <WiFiType.h>

/*
 *  Project headers.
 */
#include "actuator/actuator.h"
#include "common/global.h"
#include "common/parameter.h"
#include "common/pin.h"
#include "controller/controller.h"
#include "network/udp.h"
#include "planner/planner.h"
#include "platform/camera.h"
#include "platform/display.h"
#include "platform/io_expander.h"
#include "platform/neopixel.h"
#include "platform/serial.h"
#include "platform/wifi.h"
#include "sensor/sensor.h"
#include "task/interrupt.h"
#include "task/task.h"

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
bestEffortTask()
{
    /*
     *  Declare planner stage and set to -1.
     */
    int planner_stage = -1;

    /*
     *  Print to the OLED display using the Display class in the display header. Pass the desired line
     *  number (zero-based) as the constructor argument, and then use the class member insertion
     *  operator (<<) to stream items (string literals, variables) directly to the constructor call,
     *  similar to using the C++ STL std::cout. Note that a line exceeding the width of the display
     *  gets cut off at the display boundary and will not wrap to the next line.
     *
     *  Refer to the C++ STL ostream usages here:
     *  https://cplusplus.com/reference/ostream/ostream/operator%3C%3C/
     *
     *  Using the Display class in the display header, print to the first line of the OLED display the
     *  string literal "Biped: #" followed by the Biped serial number global variable.
     *
     *  Note that the line numbers are zero-based.
     *
     *  Refer to the display header for the display functions and the global header for the global
     *  variables.
     *
     *  TODO LAB 1 YOUR CODE HERE.
     */
    Display(0) << "Biped: #" << serial_number_;
    /*
     *  Using the Display class in the display header, print to the second line of the OLED display the
     *  string literal "Real-Time: ", followed by the real-time task execution time global variable,
     *  followed by a space, and then followed by the real-time task interval global variable.
     *
     *  Note that the line numbers are zero-based.
     *
     *  Refer to the display header for the display functions and the global header for the global
     *  variables.
     *
     *  TODO LAB 1 YOUR CODE HERE.
     */
    Display(1) << "Real-Time: " << execution_time_real_time_task_ << " "
            << interval_real_time_task_;
    /*
     *  Using the ESP-IDF ESP object in the Esp header, calculate the heap utilization percentage (i.e.,
     *  the percentage of the heap used) using the ESP-IDF getFreeHeap and getHeapSize functions in the
     *  Esp header.
     *
     *  Using the Display class in the display header, print to the third line of the OLED display the
     *  string literal "Heap: ", followed by the heap utilization percentage calculated above,
     *  and then followed by a percent sign.
     *
     *  Note that the line numbers are zero-based.
     *
     *  Refer to the display header for the display functions and the global header for the global
     *  variables.
     *
     *  TODO LAB 1 YOUR CODE HERE.
     */
    uint32_t heap_utilization_percentage = (uint32_t) (((float) ESP.getHeapSize()
            - (float) ESP.getFreeHeap()) / (float) ESP.getHeapSize() * 100.0);
    Display(2) << "Heap: " << uint32_t(heap_utilization_percentage) << "%";
    /*
     *  If the Wi-Fi global shared pointer is not a null pointer, check the Wi-Fi status using the Wi-Fi
     *  global shared pointer.
     *
     *  If the Wi-Fi status is connected (i.e., WL_CONNECTED defined in wl_status_t enum in the Arduino WiFiType
     *  header), using the Display class in the display header, print to the fourth line of the OLED display
     *  the string literal "Wi-Fi: ", followed by the Wi-Fi local IP address, obtained using the Wi-Fi global
     *  shared pointer.
     *
     *  Otherwise, using the Display class in the display header, print to the fourth line of the OLED
     *  display the string literal "Wi-Fi: disconnected".
     *
     *  Note that the line numbers are zero-based.
     *
     *  Refer to the Wi-Fi header for the Wi-Fi functions, the Arduino WiFiType header for the Wi-Fi status
     *  enum, and the global header for the global variables.
     *
     *  TODO LAB 1 YOUR CODE HERE.
     */
    if (wifi_ != nullptr)
    {
        if (wifi_->getWiFiStatus() == 3)
        {
            Display(3) << "Wi-Fi: " << wifi_->getWiFiLocalIP();
        }
        else
        {
            Display(3) << "Wi-Fi: disconnected";
        }
    }
    else
    {
        Display(3) << "Wi-Fi: disconnected";
    }

    /*
     *  If the controller global shared pointer is not a null pointer, check the controller active status
     *  using the controller global shared pointer.
     *
     *  If the controller status is active, using the Display class in the display header, print to the fifth
     *  line of the OLED display the string literal "Controller: active".
     *
     *  Otherwise, using the Display class in the display header, print to the fifth line of the OLED
     *  display the string literal "Controller: inactive".
     *
     *  Note that the line numbers are zero-based.
     *
     *  Refer to the controller header for the controller functions and the global header for the global
     *  variables.
     *
     *  TODO LAB 1 YOUR CODE HERE.
     */
    if (controller_ != nullptr)
    {
        if (controller_->getActiveStatus())
        {
            Display(4) << "Controller: active";
        }
        else
        {
            Display(4) << "Controller: inactive";
        }
    }
    else
    {
        Display(4) << "Controller: inactive";
    }



    /*
     *  If the planner global shared pointer is not a null pointer, using the planner global shared pointer,
     *  execute the plan and store return value to the planner stage local variable.
     *
     *  Refer to the planner header for the planner pure virtual functions and the global header for the
     *  global variables.
     *
     *  TODO LAB 8 YOUR CODE HERE.
     */

    /*
     *  If the planner stage local variable is less than 0, using the Display class in the display header,
     *  print to the sixth line of the OLED display the string literal "Planner: inactive".
     *
     *  Otherwise, using the Display class in the display header, print to the sixth line of the OLED
     *  display the string literal "Planner: stage " followed by the planner stage local variable.
     *
     *  Note that the line numbers are zero-based.
     *
     *  TODO LAB 1 YOUR CODE HERE.
     */
    if (planner_stage < 0)
    {
        Display(5) << "Planner: inactive";
    }
    else
    {
        Display(5) << "Planner: stage " << planner_stage;
    }

    //TODO REMOVE; demo for lab 4
    Display(6) << "Buttons: " << io_expander_a_->digitalReadPortA(static_cast<uint8_t>(IOExpanderAPortAPin::push_button_a))
            << " " << io_expander_a_->digitalReadPortA(static_cast<uint8_t>(IOExpanderAPortAPin::push_button_b))
            << " " << io_expander_a_->digitalReadPortB(static_cast<uint8_t>(IOExpanderAPortBPin::push_button_c))
            << " " << io_expander_b_->digitalReadPortA(static_cast<uint8_t>(IOExpanderBPortAPin::test_1));
    /*
     *  If the NeoPixel global shared pointer is not a null pointer, using the NeoPixel global shared
     *  pointer, show the NeoPixel frame by flushing the frame to the NeoPixel array.
     *
     *  Refer to the NeoPixel header for the NeoPixel functions and the global header for the
     *  global variables.
     *
     *  TODO LAB 1 YOUR CODE HERE.
     */
    if (neopixel_ != nullptr)
    {
        neopixel_->show();
    }
    /*
     *  Using the Display class in the display header, display the streamed items by flushing the
     *  display driver buffer to the display.
     *
     *  Refer to the display header for the display functions.
     *
     *  TODO LAB 1 YOUR CODE HERE.
     */

    Display::display();
}

void
ioExpanderAInterruptServiceTask(void* pvParameters)
{
    for (;;)
    {
        /*
         *  Sleep until woken using the FreeRTOS ulTaskNotifyTake function. Set clear count on exit
         *  (either pdFALSE or pdTRUE defined in the FreeRTOS projdefs header) such that the
         *  notification value acts like a binary semaphore. Set the maximum task wait time to be
         *  maximum delay (i.e., portMAX_DELAY defined in the FreeRTOS portmacro header).
         *
         *  Refer to the FreeRTOS projdefs header for the project definition macros, the FreeRTOS
         *  portmacro header for the port macros, and the FreeRTOS task header for the task functions.
         *
         *  TODO LAB 3 YOUR CODE HERE.
         */

        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        /*
         *  If the I/O expander A global shared pointer is not a null pointer, using the I/O expander
         *  A global shared pointer, call the I/O expander interrupt callback function.
         *
         *  Refer to the I/O expander header for the I/O expander functions.
         *
         *  TODO LAB 4 YOUR CODE HERE.
         */
        if (io_expander_a_ != nullptr)
        {
            io_expander_a_->onInterrupt();
        }

        /*
         *  The I/O expander A interrupt handler has been detached by itself. Using the attachInterrupt
         *  function in the interrupt header, re-attach the I/O expander A interrupt handler in on-high mode.
         *
         *  Note that the attachInterrupt function in the interrupt header has identical
         *  name to the Arduino attachInterrupt function. To avoid collision, explicitly resolve
         *  the namespaces by calling the attachInterrupt function in the interrupt header as
         *  biped::firmware::attachInterrupt. The explicit namespace resolution is always necessary
         *  for biped::firmware::attachInterrupt since its function signature is too similar to that
         *  of the Arduino attachInterrupt function.
         *
         *  Additionally, always wrap the ESP32 interrupt pins around the digitalPinToInterrupt
         *  macro to ensure that the given pin numbers are within the ESP32 interrupt pin range.
         *
         *  Refer to the interrupt header for the biped::firmware::attachInterrupt
         *  function, the pin header for the I/O expander interrupt pins, and the esp32-hal-gpio
         *  header for the available interrupt modes (Interrupt Modes).
         *
         *  TODO LAB 4 YOUR CODE HERE.
         */
        biped::firmware::attachInterrupt(digitalPinToInterrupt(ESP32Pin::io_expander_a_interrupt),
                ioExpanderAInterruptHandler, ONHIGH);
    }

    /*
     *  Delete this task upon exit using the FreeRTOS vTaskDelete function.
     *
     *  Note that the task handle pointer is not necessary if the task is
     *  deleting itself, i.e., task handle pointer should be a null pointer.
     *
     *  Note that due to the existence of smart pointers in C++ (unique pointer,
     *  shared pointer, etc.) One should always use the nullptr keyword to denote
     *  all null pointer values when programming in C++, instead of the C macro NULL.
     *
     *  Refer to the FreeRTOS task header for the FreeRTOS task functions.
     *
     *  TODO LAB 3 YOUR CODE HERE.
     */
    vTaskDelete(nullptr);
}

void
ioExpanderBInterruptServiceTask(void* pvParameters)
{
    for (;;)
    {
        /*
         *  Sleep until woken using the FreeRTOS ulTaskNotifyTake function. Set clear count on exit
         *  (either pdFALSE or pdTRUE defined in the FreeRTOS projdefs header) such that the
         *  notification value acts like a binary semaphore. Set the maximum task wait time to be
         *  maximum delay (i.e., portMAX_DELAY defined in the FreeRTOS portmacro header).
         *
         *  Refer to the FreeRTOS projdefs header for the project definition macros, the FreeRTOS
         *  portmacro header for the port macros, and the FreeRTOS task header for the task functions.
         *
         *  TODO LAB 3 YOUR CODE HERE.
         */
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        /*
         *  If the I/O expander B global shared pointer is not a null pointer, using the I/O expander
         *  B global shared pointer, call the I/O expander interrupt callback function.
         *
         *  Refer to the I/O expander header for the I/O expander functions.
         *
         *  TODO LAB 4 YOUR CODE HERE.
         */
        if (io_expander_b_ != nullptr)
        {
            io_expander_b_->onInterrupt();
        }

        /*
         *  The I/O expander B interrupt handler has been detached by itself. Using the attachInterrupt
         *  function in the interrupt header, re-attach the I/O expander B interrupt handler in on-high mode.
         *
         *  Note that the attachInterrupt function in the interrupt header has identical
         *  name to the Arduino attachInterrupt function. To avoid collision, explicitly resolve
         *  the namespaces by calling the attachInterrupt function in the interrupt header as
         *  biped::firmware::attachInterrupt. The explicit namespace resolution is always necessary
         *  for biped::firmware::attachInterrupt since its function signature is too similar to that
         *  of the Arduino attachInterrupt function.
         *
         *  Additionally, always wrap the ESP32 interrupt pins around the digitalPinToInterrupt
         *  macro to ensure that the given pin numbers are within the ESP32 interrupt pin range.
         *
         *  Refer to the interrupt header for the biped::firmware::attachInterrupt
         *  function, the pin header for the I/O expander interrupt pins, and the esp32-hal-gpio
         *  header for the available interrupt modes (Interrupt Modes).
         *
         *  TODO LAB 4 YOUR CODE HERE.
         */
        biped::firmware::attachInterrupt(digitalPinToInterrupt(ESP32Pin::io_expander_b_interrupt),
                ioExpanderBInterruptHandler, ONHIGH);
    }

    /*
     *  Delete this task upon exit using the FreeRTOS vTaskDelete function.
     *
     *  Note that the task handle pointer is not necessary if the task is
     *  deleting itself, i.e., task handle pointer should be a null pointer.
     *
     *  Note that due to the existence of smart pointers in C++ (unique pointer,
     *  shared pointer, etc.) One should always use the nullptr keyword to denote
     *  all null pointer values when programming in C++, instead of the C macro NULL.
     *
     *  Refer to the FreeRTOS task header for the FreeRTOS task functions.
     *
     *  TODO LAB 3 YOUR CODE HERE.
     */
    vTaskDelete(nullptr);
}

void
networkTask(void* pvParameters)
{
    /*
     *  If the Wi-Fi global shared pointer is not a null pointer, using the Wi-Fi global
     *  shared pointer, initialize the Wi-Fi driver.
     *
     *  Refer to the Wi-Fi header for the Wi-Fi functions.
     *
     *  TODO LAB 5 YOUR CODE HERE.
     */
    if (wifi_ != nullptr)
    {
        wifi_->initialize();
    }


    /*
     *  Use a spin lock to block this task when the Wi-Fi global shared pointer is a null
     *  pointer, or when the Wi-Fi status is not connected (i.e., not WL_CONNECTED defined
     *  in wl_status_t enum in the Arduino WiFiType header).
     *
     *  Refer to the Wi-Fi header for the Wi-Fi functions, the Arduino WiFiType header for
     *  the Wi-Fi status enum, and the global header for the global variables.
     *
     *  TODO LAB 5 YOUR CODE HERE.
     */
    while (wifi_ == nullptr || wifi_->getWiFiStatus() != 3)
    {
        vTaskDelay(100 / portTICK_PERIOD_MS);
    }

    /*
     *  Validate the Biped message UDP global shared pointer.
     */
    if (udp_biped_message_)
    {
        /*
         *  Using the Biped message UDP global shared pointer, initialize the UDP interface.
         *
         *  Refer to the UDP header for the UDP functions, the parameter header for the UDP
         *  ports, and the global header for the global variables.
         *
         *  TODO LAB 5 YOUR CODE HERE.
         */
        udp_biped_message_->initialize(NetworkParameter::port_udp_biped_message);

        /*
         *  Using the FreeRTOS xTaskNotifyGive function, wake up the Biped message UDP
         *  read and write tasks.
         *
         *  Refer to the FreeRTOS task header for the FreeRTOS task functions, and the global
         *  header for the global variables.
         *
         *  TODO LAB 3 YOUR CODE HERE.
         */
        xTaskNotifyGive(task_handle_udp_read_biped_message_);
        xTaskNotifyGive(task_handle_udp_write_biped_message_);
    }

    if (udp_camera_)
    {
        /*
         *  Using the camera UDP global shared pointer, initialize the UDP interface.
         *
         *  Refer to the UDP header for the UDP functions, the parameter header for the UDP
         *  ports, and the global header for the global variables.
         *
         *  TODO LAB 5 YOUR CODE HERE.
         */
        udp_camera_->initialize(NetworkParameter::port_udp_camera);

        /*
         *  Using the FreeRTOS xTaskNotifyGive function, wake up the camera UDP write task.
         *
         *  Refer to the FreeRTOS task header for the FreeRTOS task functions, and the global
         *  header for the global variables.
         *
         *  TODO LAB 3 YOUR CODE HERE.
         */
        xTaskNotifyGive(task_handle_udp_write_camera_);
    }

    /*
     *  Delete this task upon exit using the FreeRTOS vTaskDelete function.
     *
     *  Note that the task handle pointer is not necessary if the task is
     *  deleting itself, i.e., task handle pointer should be a null pointer.
     *
     *  Note that due to the existence of smart pointers in C++ (unique pointer,
     *  shared pointer, etc.) One should always use the nullptr keyword to denote
     *  all null pointer values when programming in C++, instead of the C macro NULL.
     *
     *  Refer to the FreeRTOS task header for the FreeRTOS task functions.
     *
     *  TODO LAB 3 YOUR CODE HERE.
     */
    vTaskDelete(nullptr);
}

void
realTimeTask(void* pvParameters)
{
    /*
     *  Declare start time point and set to 0.
     */
    unsigned long time_point_start = 0;

    for (;;)
    {
        /*
         *  Sleep until woken using the FreeRTOS ulTaskNotifyTake function. Set clear count on exit
         *  (either pdFALSE or pdTRUE defined in the FreeRTOS projdefs header) such that the
         *  notification value acts like a binary semaphore. Set the maximum task wait time to be
         *  maximum delay (i.e., portMAX_DELAY defined in the FreeRTOS portmacro header).
         *
         *  Refer to the FreeRTOS projdefs header for the project definition macros, the FreeRTOS
         *  portmacro header for the port macros, and the FreeRTOS task header for the task functions.
         *
         *  TODO LAB 3 YOUR CODE HERE.
         */
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        /*
         *  Using the Arduino micros timing function, update the start time point local variable to
         *  the current time in microseconds.
         *
         *  TODO LAB 3 YOUR CODE HERE.
         */
        time_point_start = micros();

        /*
         *  If the sensor global shared pointer is not a null pointer, using the sensor global shared
         *  pointer, perform fast domain sensing.
         *
         *  Refer to the global header for the global variables and the sensor header for the sensor
         *  functions.
         *
         *  TODO LAB 6 YOUR CODE HERE.
         */

        /*
         *  If the controller global shared pointer is not a null pointer, using the controller global
         *  shared pointer, perform fast domain control.
         *
         *  Refer to the global header for the global variables and the controller header for the
         *  controller functions.
         *
         *  TODO LAB 7 YOUR CODE HERE.
         */

        /*
         *  Slow domain tasks.
         */
        if (timer_domain_ >= PeriodParameter::slow)
        {
            /*
             *  If the sensor global shared pointer is not a null pointer, using the sensor global shared
             *  pointer, perform slow domain sensing.
             *
             *  Refer to the global header for the global variables and the sensor header for the sensor
             *  functions.
             *
             *  TODO LAB 6 YOUR CODE HERE.
             */

            /*
             *  If the controller global shared pointer is not a null pointer, using the controller global
             *  shared pointer, perform slow domain control.
             *
             *  Refer to the global header for the global variables and the controller header for the
             *  controller functions.
             *
             *  TODO LAB 7 YOUR CODE HERE.
             */

            /*
             *  Reset period domain timer global variable to 0.
             *
             *  Refer to the global header for the global variables.
             *
             *  TODO LAB 6 YOUR CODE HERE.
             */
        }

        /*
         *  If the controller global shared pointer is not a null pointer, using the controller global
         *  shared pointer, obtain the actuation command struct.
         *
         *  If the actuator global shared pointer is not a null pointer, using the actuator global
         *  shared pointer, perform actuation using the obtained actuation command struct.
         *
         *  Refer to the global header for the global variables, the controller header for the
         *  controller functions, and the actuator header for the actuator functions.
         *
         *  TODO LAB 7 YOUR CODE HERE.
         */

        /*
         *  Add the fast domain period to the period domain timer global variable.
         *
         *  Refer to the global header for the global variables and the parameter header for the
         *  period parameters.
         *
         *  TODO LAB 6 YOUR CODE HERE.
         */

        /*
         *  Calculate the real-time task execution time by subtracting the current time in microseconds,
         *  obtained using the Arduino micros timing function, with the start time point local variable
         *  and storing the result in the real-time task execution time global variable.
         *
         *  Refer to the global header for the global variables.
         *
         *  TODO LAB 3 YOUR CODE HERE.
         */
        unsigned long time_point_current = micros();
        execution_time_real_time_task_ = time_point_current - time_point_start;
    }

    /*
     *  Delete this task upon exit using the FreeRTOS vTaskDelete function.
     *
     *  Note that the task handle pointer is not necessary if the task is
     *  deleting itself, i.e., task handle pointer should be a null pointer.
     *
     *  Note that due to the existence of smart pointers in C++ (unique pointer,
     *  shared pointer, etc.) One should always use the nullptr keyword to denote
     *  all null pointer values when programming in C++, instead of the C macro NULL.
     *
     *  Refer to the FreeRTOS task header for the FreeRTOS task functions.
     *
     *  TODO LAB 3 YOUR CODE HERE.
     */
    vTaskDelete(nullptr);
}

void
udpReadBipedMessageTask(void* pvParameters)
{
    /*
     *  Sleep until woken using the FreeRTOS ulTaskNotifyTake function. Set clear count on exit
     *  (either pdFALSE or pdTRUE defined in the FreeRTOS projdefs header) such that the
     *  notification value acts like a binary semaphore. Set the maximum task wait time to be
     *  maximum delay (i.e., portMAX_DELAY defined in the FreeRTOS portmacro header).
     *
     *  Refer to the FreeRTOS projdefs header for the project definition macros, the FreeRTOS
     *  portmacro header for the port macros, and the FreeRTOS task header for the task functions.
     *
     *  TODO LAB 3 YOUR CODE HERE.
     */
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    /*
     *  Task loop.
     */
    for (;;)
    {
        /*
         *  Skip the current iteration if the Wi-Fi global shared pointer is a null pointer or the
         *  Wi-Fi status is not connected (i.e., not WL_CONNECTED defined in wl_status_t enum in the
         *  Arduino WiFiType header).
         *
         *  Refer to the Wi-Fi header for the Wi-Fi functions, the Arduino WiFiType header for
         *  the Wi-Fi status enum, and the global header for the global variables.
         *
         *  TODO LAB 5 YOUR CODE HERE.
         */
        if (wifi_ == nullptr || wifi_->getWiFiStatus() != 3)
        {
            vTaskDelay(100 / portTICK_PERIOD_MS);
            continue;
        }

        /*
         *  Using the Biped message UDP global shared pointer, read the message from the Biped
         *  ground station into a string.
         *
         *  Refer to the UDP header for the UDP functions, the parameter header for the network
         *  parameters, and the global header for the global variables.
         *
         *  TODO LAB 5 YOUR CODE HERE.
         */
        std::string message = udp_biped_message_->read(NetworkParameter::ip_ground_station, NetworkParameter::port_udp_biped_message, NetworkParameter::buffer_size_biped_message);

        /*
         *  Skip the current iteration if the message read above is empty.
         *
         *  TODO LAB 5 YOUR CODE HERE.
         */
        if (message.empty())
        {
            continue;
        }

        /*
         *  Declare deserialized Biped message struct, serialized message buffer,
         *  and the deserializer.
         */
        BipedMessage message_deserialized;
        std::vector<unsigned char> message_serialized(message.begin(), message.end());
        zpp::serializer::memory_input_archive deserializer(message_serialized);

        /*
         *  Deserialize the serialized message buffer into the Biped message struct.
         */
        const auto result = deserializer(message_deserialized);

        /*
         *  Print warning message to serial and skip the current iteration upon
         *  deserialization failure.
         */
        if (!result)
        {
            Serial(LogLevel::warn) << "Failed to deserialize Biped message.";
            continue;
        }

        /*
         *  If the controller global shared pointer is not a null pointer, using the controller
         *  global shared pointer, set the controller parameter and controller reference structs
         *  in the deserialized Biped message struct to the controller.
         *
         *  Refer to the controller header for the controller functions, the type header for Biped
         *  message struct entries, and the global header for the global variables.
         *
         *  TODO LAB 5 YOUR CODE HERE.
         */
        if (controller_ != nullptr)
        {
            controller_->setControllerParameter(message_deserialized.controller_parameter);
            controller_->setControllerReference(message_deserialized.controller_reference);
        }
    }

    /*
     *  Delete this task upon exit using the FreeRTOS vTaskDelete function.
     *
     *  Note that the task handle pointer is not necessary if the task is
     *  deleting itself, i.e., task handle pointer should be a null pointer.
     *
     *  Note that due to the existence of smart pointers in C++ (unique pointer,
     *  shared pointer, etc.) One should always use the nullptr keyword to denote
     *  all null pointer values when programming in C++, instead of the C macro NULL.
     *
     *  Refer to the FreeRTOS task header for the FreeRTOS task functions.
     *
     *  TODO LAB 3 YOUR CODE HERE.
     */
    vTaskDelete(nullptr);
}

void
udpWriteBipedMessageTask(void* pvParameters)
{
    /*
     *  Sleep until woken using the FreeRTOS ulTaskNotifyTake function. Set clear count on exit
     *  (either pdFALSE or pdTRUE defined in the FreeRTOS projdefs header) such that the
     *  notification value acts like a binary semaphore. Set the maximum task wait time to be
     *  maximum delay (i.e., portMAX_DELAY defined in the FreeRTOS portmacro header).
     *
     *  Refer to the FreeRTOS projdefs header for the project definition macros, the FreeRTOS
     *  portmacro header for the port macros, and the FreeRTOS task header for the task functions.
     *
     *  TODO LAB 3 YOUR CODE HERE.
     */
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    /*
     *  Task loop.
     */
    for (;;)
    {
        /*
         *  Skip the current iteration if the Wi-Fi global shared pointer is a null pointer or the
         *  Wi-Fi status is not connected (i.e., not WL_CONNECTED defined in wl_status_t enum in the
         *  Arduino WiFiType header).
         *
         *  Refer to the Wi-Fi header for the Wi-Fi functions, the Arduino WiFiType header for
         *  the Wi-Fi status enum, and the global header for the global variables.
         *
         *  TODO LAB 5 YOUR CODE HERE.
         */
        if (wifi_ == nullptr || wifi_->getWiFiStatus() != 3)
        {
            vTaskDelay(100 / portTICK_PERIOD_MS);
            continue;
        }

        /*
         *  Declare Biped message struct, serialized message buffer, static message sequence,
         *  and the serializer.
         */
        BipedMessage message;
        std::vector<unsigned char> message_serialized;
        static unsigned long long sequence = 0;
        zpp::serializer::memory_output_archive serializer(message_serialized);

        /*
         *  Set the sequence in the Biped message struct as the static message sequence local
         *  variable and the timestamp in the Biped message struct local variable as the current
         *  time in microseconds, obtained using the Arduino micros timing function.
         *
         *  TODO LAB 5 YOUR CODE HERE.
         */
        message.sequence = sequence;
        message.timestamp = micros();

        /*
         *  If the sensor global shared pointer is not a null pointer, using the sensor global
         *  shared pointer, obtain the encoder, IMU, and time-of-flight data structs and set
         *  them to the encoder, IMU, and time-of-flight data structs in the Biped message struct
         *  local variable.
         *
         *  Refer to the sensor header for the sensor functions, the type header for Biped
         *  message struct entries, and the global header for the global variables.
         *
         *  TODO LAB 5 YOUR CODE HERE.
         */
        if (sensor_ != nullptr)
        {
            message.encoder_data = sensor_->getEncoderData();
            message.imu_data = sensor_->getIMUData();
            message.time_of_flight_data = sensor_->getTimeOfFlightData();
        }

        /*
         *  If the controller global shared pointer is not a null pointer, using the controller global
         *  shared pointer, obtain the controller parameter and controller reference structs and set
         *  them to the controller parameter and controller reference structs in the Biped message struct
         *  local variable.
         *
         *  Refer to the controller header for the controller functions, the type header for Biped
         *  message struct entries, and the global header for the global variables.
         *
         *  TODO LAB 5 YOUR CODE HERE.
         */
        if (controller_ != nullptr)
        {
            message.controller_parameter = controller_->getControllerParameter();
            message.controller_reference = controller_->getControllerReference();
        }

        /*
         *  If the actuator global shared pointer is not a null pointer, using the actuator global
         *  shared pointer, obtain the actuation command struct and set it to the actuation command
         *  struct in the Biped message struct local variable.
         *
         *  Refer to the actuator header for the actuator functions, the type header for Biped
         *  message struct entries, and the global header for the global variables.
         *
         *  TODO LAB 5 YOUR CODE HERE.
         */
        if (actuator_ != nullptr)
        {
            message.actuation_command = actuator_->getActuationCommand();
        }

        /*
         *  Serialize the Biped message struct into the serialized message buffer.
         */
        const auto result = serializer(message);

        if (result)
        {
            /*
             *  Using the Biped message UDP global shared pointer, write the serialized message buffer
             *  to the Biped ground station.
             *
             *  Note that the serialized message buffer can be converted into a string using its iterator.
             *
             *  Refer to the UDP header for the UDP functions, the parameter header for the network
             *  parameters, and the global header for the global variables.
             *
             *  TODO LAB 5 YOUR CODE HERE.
             */
            std::string message_serialized_str(message_serialized.begin(), message_serialized.end());
            udp_biped_message_->write(NetworkParameter::ip_ground_station, NetworkParameter::port_udp_biped_message, message_serialized_str);
        }
        else
        {
            /*
             *  Print warning message to serial upon serialization failure. Otherwise,
             */
            Serial(LogLevel::warn) << "Failed to serialize Biped message.";
        }

        /*
         *  Increment the static message sequence local variable.
         *
         *  TODO LAB 5 YOUR CODE HERE.
         */
        sequence++;
    }

    /*
     *  Delete this task upon exit using the FreeRTOS vTaskDelete function.
     *
     *  Note that the task handle pointer is not necessary if the task is
     *  deleting itself, i.e., task handle pointer should be a null pointer.
     *
     *  Note that due to the existence of smart pointers in C++ (unique pointer,
     *  shared pointer, etc.) One should always use the nullptr keyword to denote
     *  all null pointer values when programming in C++, instead of the C macro NULL.
     *
     *  Refer to the FreeRTOS task header for the FreeRTOS task functions.
     *
     *  TODO LAB 3 YOUR CODE HERE.
     */
    vTaskDelete(nullptr);
}

void
udpWriteCameraTask(void* pvParameters)
{
    /*
     *  Sleep until woken using the FreeRTOS ulTaskNotifyTake function. Set clear count on exit
     *  (either pdFALSE or pdTRUE defined in the FreeRTOS projdefs header) such that the
     *  notification value acts like a binary semaphore. Set the maximum task wait time to be
     *  maximum delay (i.e., portMAX_DELAY defined in the FreeRTOS portmacro header).
     *
     *  Refer to the FreeRTOS projdefs header for the project definition macros, the FreeRTOS
     *  portmacro header for the port macros, and the FreeRTOS task header for the task functions.
     *
     *  TODO LAB 3 YOUR CODE HERE.
     */
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    /*
     *  Task loop.
     */
    for (;;)
    {
        /*
         *  If the camera global shared pointer is not a null pointer, using the camera global
         *  shared pointer, send the camera frame over UDP to the Biped ground station via the
         *  camera UDP global shared pointer.
         *
         *  Refer to the global header for the global variables and the parameter header for the
         *  IP addresses and UDP ports.
         *
         *  TODO LAB 5 YOUR CODE HERE.
         */
        if (camera_ != nullptr)
        {
            camera_->SendJPGFrameOverUDP(udp_camera_, NetworkParameter::ip_ground_station, NetworkParameter::port_udp_camera);
        }
    }

    /*
     *  Delete this task upon exit using the FreeRTOS vTaskDelete function.
     *
     *  Note that the task handle pointer is not necessary if the task is
     *  deleting itself, i.e., task handle pointer should be a null pointer.
     *
     *  Note that due to the existence of smart pointers in C++ (unique pointer,
     *  shared pointer, etc.) One should always use the nullptr keyword to denote
     *  all null pointer values when programming in C++, instead of the C macro NULL.
     *
     *  Refer to the FreeRTOS task header for the FreeRTOS task functions.
     *
     *  TODO LAB 3 YOUR CODE HERE.
     */
    vTaskDelete(nullptr);
}
}   // namespace firmware
}   // namespace biped
