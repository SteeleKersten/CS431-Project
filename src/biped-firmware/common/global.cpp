/**
 *  @file   global.cpp
 *  @author Simon Yu
 *  @date   12/01/2022
 *  @brief  Global variable source.
 *
 *  This file initializes the global variables.
 */

/*
 *  Project headers.
 */
#include "common/global.h"

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
 *  Initialize shared pointers to null pointers.
 */
std::shared_ptr<Actuator> actuator_ = nullptr;
std::shared_ptr<Camera> camera_ = nullptr;
std::shared_ptr<Controller> controller_ = nullptr;
std::shared_ptr<IOExpander> io_expander_a_ = nullptr;
std::shared_ptr<IOExpander> io_expander_b_ = nullptr;
std::shared_ptr<NeoPixel> neopixel_ = nullptr;
std::shared_ptr<Planner> planner_ = nullptr;
std::shared_ptr<Sensor> sensor_ = nullptr;
std::shared_ptr<Timer> timer_ = nullptr;
std::shared_ptr<UDP> udp_biped_message_ = nullptr;
std::shared_ptr<UDP> udp_camera_ = nullptr;
std::shared_ptr<WiFi> wifi_ = nullptr;

/*
 *  Initialize I2C driver mutex and mutex lock.
 */
std::mutex mutex_wire_;
std::unique_lock<std::mutex> lock_wire_ = std::unique_lock<std::mutex>(mutex_wire_,
        std::defer_lock);

/*
 *  Initialize task handles to null pointers.
 */
TaskHandle_t task_handle_io_expander_a_interrupt_service_ = nullptr;
TaskHandle_t task_handle_io_expander_b_interrupt_service_ = nullptr;
TaskHandle_t task_handle_network_ = nullptr;
TaskHandle_t task_handle_real_time_ = nullptr;
TaskHandle_t task_handle_udp_read_biped_message_ = nullptr;
TaskHandle_t task_handle_udp_write_biped_message_ = nullptr;
TaskHandle_t task_handle_udp_write_camera_ = nullptr;

/*
 *  Initialize timing variables to 0.
 */
unsigned long execution_time_real_time_task_ = 0;
unsigned long interval_real_time_task_ = 0;
double timer_domain_ = 0;

/*
 *  Initialize Biped serial number to 0.
 */
unsigned serial_number_ = 0;
}   // namespace firmware
}   // namespace biped
