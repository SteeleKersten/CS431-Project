/**
 *  @file   global.h
 *  @author Simon Yu
 *  @date   12/01/2022
 *  @brief  Global variable header.
 *
 *  This file defines the global variables.
 */

/*
 *  Include guard.
 */
#ifndef COMMON_GLOBAL_H_
#define COMMON_GLOBAL_H_

/*
 *  External headers.
 */
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <memory>
#include <mutex>

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
/*
 *  Forward declarations.
 */
class Actuator;
class Camera;
class Controller;
class IOExpander;
class NeoPixel;
class Planner;
class Sensor;
class Timer;
class UDP;
class WiFi;

extern std::shared_ptr<Actuator> actuator_; //!< Actuator shared pointer.
extern std::shared_ptr<Camera> camera_; //!< Camera shared pointer.
extern std::shared_ptr<Controller> controller_; //!< Controller shared pointer.
extern std::shared_ptr<IOExpander> io_expander_a_;  //!< I/O expander A shared pointer.
extern std::shared_ptr<IOExpander> io_expander_b_;  //!< I/O expander B shared pointer.
extern std::shared_ptr<NeoPixel> neopixel_; //!< NeoPixel shared pointer.
extern std::shared_ptr<Planner> planner_;   //!< Planner shared pointer.
extern std::shared_ptr<Sensor> sensor_; //!< Sensor shared pointer.
extern std::shared_ptr<Timer> timer_; //!< Timer shared pointer.
extern std::shared_ptr<UDP> udp_biped_message_; //!< Biped message UDP shared pointer.
extern std::shared_ptr<UDP> udp_camera_; //!< Camera UDP shared pointer.
extern std::shared_ptr<WiFi> wifi_; //!< Wi-Fi shared pointer.

extern std::mutex mutex_wire_;  //!< I2C driver object mutex.
extern std::unique_lock<std::mutex> lock_wire_; //!< I2C driver object mutex lock.

extern TaskHandle_t task_handle_io_expander_a_interrupt_service_; //!< I/O expander A interrupt service task handle.
extern TaskHandle_t task_handle_io_expander_b_interrupt_service_; //!< I/O expander B interrupt service task handle.
extern TaskHandle_t task_handle_network_;  //!< Network task handle.
extern TaskHandle_t task_handle_real_time_; //!< Real-time task handle.
extern TaskHandle_t task_handle_udp_read_biped_message_;    //!< Biped message UDP read task handle.
extern TaskHandle_t task_handle_udp_write_biped_message_;    //!< Biped message UDP write task handle.
extern TaskHandle_t task_handle_udp_write_camera_;    //!< Camera UDP write task handle.

/*
 *  The period domain timer below is used in the real-time task
 *  function for determining whether a full slow domain period
 *  has passed since the last reset of the timer. The slow domain
 *  period is a multiple of fast domain period.
 */
extern unsigned long execution_time_real_time_task_; //!< Real-time task execution time, in microseconds.
extern unsigned long interval_real_time_task_;  //!< Real-time task interval, in microseconds.
extern double timer_domain_;    //!< Period domain timer, in seconds.

extern unsigned serial_number_; //!< Biped serial number.
}   // namespace firmware
}   // namespace biped

#endif  // COMMON_GLOBAL_H_
