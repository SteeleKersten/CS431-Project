/**
 *  @file   controller.cpp
 *  @author Simon Yu
 *  @date   01/13/2022
 *  @brief  Controller class source.
 *
 *  This file implements the controller class.
 */

/*
 *  Project headers.
 */
#include "common/global.h"
#include "common/parameter.h"
#include "controller/controller.h"
#include "platform/display.h"
#include "sensor/sensor.h"
#include "utility/math.h"

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
Controller::Controller() : active_(false), output_position_x_(0), output_attitude_y_(0),
        output_attitude_z_(0)
{
    /*
     *  Set entries in the X position (forward/backward) PID controller gain
     *  struct in the class member controller parameter struct.
     *
     *  Tune this controller only after successfully tuning the Y attitude
     *  (pitch) controller. Before that, set all gains of this controller to 0.
     *
     *  For this controller, all gains should be positive. If your system
     *  diverges with positive gains, make sure you have all sensor data
     *  in the standard body reference frame, and you have subtracted the
     *  state with reference (Y - R), not the other way around, in the PID
     *  controller control function.
     *
     *  To tune this controller, start by increasing the differential gain
     *  first. After the controller is able to resist some velocity change
     *  (damping), then increase the proportional gain until Biped is able
     *  to track the X position reference and no longer run off when pushed
     *  along the X axis (X position control)
     *
     *  It is required to use the Biped ground station to perform iterative
     *  tuning of the controller gains. Refer to the README for this lab for
     *  detailed instructions. Once you have obtained a satisfactory set of
     *  gain values using the Biped ground station, set those gain values in
     *  this constructor so that they are compiled into the firmware.
     *
     *  However, if you were unable to get the Biped ground station to function
     *  or work reliably, with the TA's approval, perform the controller tuning
     *  by directly changing the gain values in this constructor, and re-compile
     *  and re-flash the firmware every time you modify the gain values.
     *
     *  After successfully tuning all other gains for all controllers, tune the
     *  integral gains using the Biped ground station to minimize any steady-state
     *  errors. With the Biped ground station, the steady-state errors can be
     *  easily observed using the controller response plots. Start by increasing
     *  the maximum integrated error. Then, increase the integral gain until the
     *  steady-state errors are minimized.
     *
     *  Imagine the integral control as a piggy bank. The maximum integrated error
     *  is the size of the piggy bank, and the integral gain is the speed at
     *  which you insert the coins into the piggy bank. The higher the maximum
     *  integrated error, the more dominant the integral control would become. The
     *  higher the integral gain, the faster the integrated error would reach the
     *  maximum integrated error, given a constant source of steady-state errors.
     *  However, consider what would happen if the integrated error becomes too
     *  big (this is also why PI controllers can be dangerous in certain cases).
     *  The integral gain tuning is optional if controller tuning was performed
     *  without using the Biped ground station (with the TA's approval).
     *
     *  Refer to Lecture 13 for the PID controller tuning heuristics.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
    controller_parameter_.pid_controller_gain_position_x.proportional = 0;
    controller_parameter_.pid_controller_gain_position_x.differential = 0;
    controller_parameter_.pid_controller_gain_position_x.integral = 0;
    controller_parameter_.pid_controller_gain_position_x.integral_max = 0;

    /*
     *  Set entries in the Y attitude (pitch) PID controller gain struct in
     *  the class member controller parameter struct.
     *
     *  Tune this controller first. Before you start, set all the gains of
     *  all controllers to 0, if not already, including this one.
     *
     *  For this controller, all gains should be negative (except for
     *  max integral). If your system diverges with negative gains, make
     *  sure you have all sensor data in the standard body reference frame,
     *  and you have subtracted state with reference (Y - R), not the other
     *  way around, in the PID controller control function.
     *
     *  To tune this controller, start with the proportional gain first.
     *  Increase the magnitude of the proportional gain until Biped starts
     *  mildly oscillating around the Y axis (marginally stable) when held
     *  upright. Then, start increasing the magnitude of the differential
     *  gain until any oscillations disappear (damping) when held upright.
     *
     *  Note that at this point, it is perfectly normal if your Biped is
     *  still not able to balance itself and tries to run off along the
     *  X axis, as the pitch controller alone cannot hold the Biped in
     *  place. That would be the job for the X position controller
     *  (the next one you should be tuning).
     *
     *  It is required to use the Biped ground station to perform iterative
     *  tuning of the controller gains. Refer to the README for this lab for
     *  detailed instructions. Once you have obtained a satisfactory set of
     *  gain values using the Biped ground station, set those gain values in
     *  this constructor so that they are compiled into the firmware.
     *
     *  However, if you were unable to get the Biped ground station to function
     *  or work reliably, with the TA's approval, perform the controller tuning
     *  by directly changing the gain values in this constructor, and re-compile
     *  and re-flash the firmware every time you modify the gain values.
     *
     *  After successfully tuning all other gains for all controllers, tune the
     *  integral gains using the Biped ground station to minimize any steady-state
     *  errors. With the Biped ground station, the steady-state errors can be
     *  easily observed using the controller response plots. Start by increasing
     *  the maximum integrated error. Then, increase the integral gain until the
     *  steady-state errors are minimized.
     *
     *  Imagine the integral control as a piggy bank. The maximum integrated error
     *  is the size of the piggy bank, and the integral gain is the speed at
     *  which you insert the coins into the piggy bank. The higher the maximum
     *  integrated error, the more dominant the integral control would become. The
     *  higher the integral gain, the faster the integrated error would reach the
     *  maximum integrated error, given a constant source of steady-state errors.
     *  However, consider what would happen if the integrated error becomes too
     *  big (this is also why PI controllers can be dangerous in certain cases).
     *  The integral gain tuning is optional if controller tuning was performed
     *  without using the Biped ground station (with the TA's approval).
     *
     *  Refer to Lecture 13 for the PID controller tuning heuristics.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
    controller_parameter_.pid_controller_gain_attitude_y.proportional = 0;
    controller_parameter_.pid_controller_gain_attitude_y.differential = 0;
    controller_parameter_.pid_controller_gain_attitude_y.integral = 0;
    controller_parameter_.pid_controller_gain_attitude_y.integral_max = 0;

    /*
     *  Set entries in the Z attitude (yaw) open-loop controller gain value
     *  and PID controller gain struct in the class member controller parameter
     *  struct.
     *
     *  Tune this controller in the end. By now, you should have successfully
     *  tuned the other two controllers. Also, you should have successfully
     *  tuned the input saturation bounds below. Before that, set all gains
     *  of this controller to 0.
     *
     *  For this controller, the open-loop gain should be positive, while the
     *  PID gains should be negative. If your system diverges, make sure you
     *  have all sensor data in the standard body reference frame, and you
     *  have subtracted state with reference (Y - R), not the other way around,
     *  in the PID controller control function.
     *
     *  To tune this controller, start with the open-loop gain first. Increase
     *  the magnitude of the open-loop gain until your Biped is able to turn right
     *  to 90 degrees under the command of the waypoint planner example plan
     *  (open-loop yaw control). Then, increase the magnitude of the
     *  differential gain until any oscillations disappear (damping).
     *
     *  Both the integral and the proportional gain should be 0 since this
     *  is an open-loop controller with differential damping.
     *
     *  Why do we use an open-loop controller instead of a closed-loop PID
     *  controller for the yaw control? The reason is that the MPU6050 does not
     *  have a compass. BMX160 does; however, the BMX160 compass data is
     *  unreliable and are prone to drift. Therefore, we currently do not have
     *  reliable yaw sensor data to control the yaw in a closed loop. Thus,
     *  we had to resort to using open-loop control for the yaw.
     *
     *  TODO LAB 8 YOUR CODE HERE.
     */
    controller_parameter_.attitude_z_gain_open_loop = 0;
    controller_parameter_.pid_controller_gain_attitude_z.proportional = 0;
    controller_parameter_.pid_controller_gain_attitude_z.differential = 0;
    controller_parameter_.pid_controller_gain_attitude_z.integral = 0;
    controller_parameter_.pid_controller_gain_attitude_z.integral_max = 0;

    /*
     *  Set entries in the X position (forward/backward) PID controller
     *  saturation struct in the class member controller parameter struct.
     *
     *  Adjust the input saturation bounds such that your Biped
     *  is able to go forward and backward with moderate speed
     *  under the command of the waypoint planner example plan.
     *
     *  The lower bound controls the forward saturation, and should
     *  be a negative value. The upper bound controls the backward
     *  saturation, and should be a positive value. The bounds, in
     *  turn, also control the velocity of Biped (Biped would fail
     *  to balance if moving too fast and would fail to overcome the
     *  disturbances (friction) if too slow. The bounds are also
     *  affected by the center of gravity of your Biped, and each
     *  Biped may have a slightly different center of gravity.
     *  Therefore, it is important to stick with the same Biped
     *  throughout the remaining labs.
     *
     *  Why does the controller input need to be saturated? Image your
     *  Biped is current at 0 meter X position, and your reference is
     *  1 meter. In this case, the magnitude of the error (e) would
     *  simply be 1. However, if your reference is instead 1000 meters,
     *  then the magnitude of the error (e) would be 1000, which would be
     *  around four times the maximum motor PWM values. Therefore, the
     *  controller input needs to be saturated within a limited range
     *  to prevent excessive control output.
     *
     *  Learn more about controller saturation here:
     *  https://www.dmi.unict.it/santoro/teaching/sr/slides/PIDSaturation.pdf
     *
     *  TODO LAB 8 YOUR CODE HERE.
     */
    controller_parameter_.pid_controller_saturation_position_x.input_lower = std::numeric_limits<
            double>::lowest();
    controller_parameter_.pid_controller_saturation_position_x.input_upper = std::numeric_limits<
            double>::max();

    /*
     *  Using the setControllerParameter class member function, set the
     *  controller parameter to be the class member controller parameter struct.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Using the setControllerReference class member function, set the
     *  controller reference to be the class member controller reference struct.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Initialize NeoPixel frame for controller active status.
     */
    neopixel_frame_active_ = std::make_shared<NeoPixel::Frame>();
    neopixel_frame_active_->push_back(Eigen::Vector3i(0, 255, 0));
    neopixel_frame_active_->push_back(Eigen::Vector3i(0, 255, 0));
    neopixel_frame_active_->push_back(Eigen::Vector3i(0, 255, 0));
    neopixel_frame_active_->push_back(Eigen::Vector3i(0, 255, 0));

    /*
     *  Initialize NeoPixel frame for controller inactive status.
     */
    neopixel_frame_inactive_ = std::make_shared<NeoPixel::Frame>();
    neopixel_frame_inactive_->push_back(Eigen::Vector3i(0, 0, 255));
    neopixel_frame_inactive_->push_back(Eigen::Vector3i(0, 0, 255));
    neopixel_frame_inactive_->push_back(Eigen::Vector3i(0, 0, 255));
    neopixel_frame_inactive_->push_back(Eigen::Vector3i(0, 0, 255));
}

ActuationCommand
Controller::getActuationCommand() const
{
    /*
     *  Return the class member actuation command struct.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
    return ActuationCommand();
}

bool
Controller::getActiveStatus() const
{
    /*
     *  Return the class member controller active flag.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
    return false;
}

ControllerParameter
Controller::getControllerParameter() const
{
    /*
     *  Return the class member controller parameter struct.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
    return ControllerParameter();
}

ControllerReference
Controller::getControllerReference() const
{
    /*
     *  Return the class member controller reference struct.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
    return ControllerReference();
}

void
Controller::setControllerParameter(const ControllerParameter& controller_parameter)
{
    /*
     *  Store the given controller parameter struct in the
     *  function parameter to the class member controller
     *  parameter struct.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Set the Z attitude open-loop controller gain entry in the class member
     *  controller parameter struct to the class member Z attitude open-loop
     *  controller.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Set the PID controller gain structs in the class member controller
     *  parameter struct to the corresponding class member PID controllers.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Set the controller saturation structs in the class member controller
     *  parameter struct to the corresponding class member controllers.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
}

void
Controller::setControllerReference(const ControllerReference& controller_reference)
{
    /*
     *  Store the given controller reference struct in the
     *  function parameter to the class member controller
     *  reference struct.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Set the entries in the class member controller reference
     *  struct to the corresponding class member controllers.
     *
     *  You should set the Z attitude (yaw) controller reference
     *  in the class member controller reference struct to both the
     *  class member Z attitude (yaw) open-loop controller and the
     *  class member Z attitude (yaw) PID controller.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
}

void
Controller::setPeriod(const double& period, const bool& fast_domain)
{
    /*
     *  The reason behind the fast and slow time domain setup here
     *  is similar to the explanation provided in the sensor header.
     *  Controlling Y attitude (pitch) requires rapid response and
     *  little to no delay, thus demanding a very short period.
     *  For controlling X position (forward/backward) and Z attitude
     *  (yaw), on the other hand, a longer period is more suitable
     *  as they are less mission-critical and their sensor data is
     *  also noisier.
     */
    if (fast_domain)
    {
        /*
         *  Set the class member Y attitude (pitch) PID controller
         *  period.
         *
         *  TODO LAB 7 YOUR CODE HERE.
         */
    }
    else
    {
        /*
         *  Set the class member X position (forward/backward)
         *  PID controller period.
         *
         *  TODO LAB 7 YOUR CODE HERE.
         */

        /*
         *  Set the class member Z attitude (yaw) PID controller
         *  period.
         *
         *  TODO LAB 7 YOUR CODE HERE.
         */
    }
}

void
Controller::control(const bool& fast_domain)
{
    /*
     *  Get the encoder and IMU data
     *  structs from the sensor object.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Update the controller active status using the IMU data struct.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  The reason behind the fast and slow time domain setup here
     *  is similar to the explanation provided in the sensor header.
     *  Controlling Y attitude (pitch) requires rapid response and
     *  little to no delay, thus demanding a very short period.
     *  For controlling X position (forward/backward) and Z attitude
     *  (yaw), on the other hand, a longer period is more suitable
     *  as they are less mission-critical and their sensor data is
     *  also noisier.
     */
    if (fast_domain)
    {
        /*
         *  Set the plant state input (Y) of the class member Y attitude
         *  (pitch) PID controller to be the Y attitude (pitch) in the
         *  IMU data struct.
         *
         *  TODO LAB 7 YOUR CODE HERE.
         */

        /*
         *  Set the error differential input (delta e) of the class member
         *  Y attitude (pitch) PID controller to be the Y angular velocity
         *  in the IMU data struct.
         *
         *  TODO LAB 7 YOUR CODE HERE.
         */

        /*
         *  Execute the class member Y attitude (pitch) PID controller
         *  and store the output into the class member Y attitude (pitch)
         *  controller output variable.
         *
         *  TODO LAB 7 YOUR CODE HERE.
         */
    }
    else
    {
        /*
         *  Set the plant state input (Y) of the class member X position
         *  (forward/backward) PID controller to be the X position in
         *  the encoder data struct.
         *
         *  TODO LAB 7 YOUR CODE HERE.
         */

        /*
         *  Set the error differential input (delta e) of the class member
         *  X position (forward/backward) PID controller to be the X
         *  linear velocity in the encoder data struct.
         *
         *  TODO LAB 7 YOUR CODE HERE.
         */

        /*
         *  Set the error differential input (delta e) of the class member
         *  Z attitude (yaw) PID controller to be the Z angular velocity in
         *  the IMU data struct.
         *
         *  TODO LAB 7 YOUR CODE HERE.
         */

        /*
         *  Execute the class member X position (forward/backward) PID
         *  controller and store the output into the class member X position
         *  (forward/backward) controller output variable.
         *
         *  TODO LAB 7 YOUR CODE HERE.
         */

        /*
         *  Execute the class member Z attitude (yaw) open-loop and PID
         *  controllers, multiply the output of the Z attitude (yaw) open-loop
         *  controller with the magnitude of the X velocity in the encoder data
         *  struct, add the output of Z attitude (yaw) PID controller to the
         *  product, and store the final output into the class member Z attitude
         *  (yaw) controller output variable.
         *
         *  The rationale behind scaling the Z attitude (yaw) open-loop controller
         *  output with the magnitude of the X velocity stems from the fact that,
         *  under open-loop yaw control, Biped struggles to make accurate and stable
         *  sharp turns at low speeds, given that the open-loop yaw control essentially
         *  functions as a non-feedback bias between the two motors. To address this,
         *  it is best to have the open-loop yaw control output be proportional to the
         *  magnitude of the X linear velocity. The velocity-based scaling ensures
         *  that the open-loop yaw control only contributes when Biped reaches a decent
         *  X linear velocity, optimizing its open-loop turning performance.
         *
         *  TODO LAB 7 YOUR CODE HERE.
         */
    }

    /*
     *  Produce the left motor output by adding all three
     *  class member controller outputs.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Produce the right motor output by adding the
     *  class member X position controller output with the
     *  class member Y attitude controller output and then
     *  subtract the class member Z attitude controller output.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  If the controller is inactive, stop the motors
     *  by setting both the motor outputs to 0.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Set the motor enable in the class member actuation
     *  command struct to true, arming the motors.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Set the motor directions in the class member actuation
     *  command struct to be the sign of the motor output values
     *  computed above. For example, the left motor forward flag
     *  should be true if the left motor output value is greater
     *  than or equal to the minimum motor PWM value.
     *
     *  Refer to the parameter header for the maximum and minimum
     *  motor PWM values.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */

    /*
     *  Using the clamp function from the math header, clamp the
     *  magnitude of the motor output values to be within the
     *  minimum and maximum motor PWM values. Remember to
     *  static_cast the motor parameter values to double before
     *  passing them to the clamp function.
     *
     *  Note that one should always use C++ explicit type casts
     *  (static_cast, dynamic_cast, etc.) when programming in C++
     *  instead of the C-style type cast, as the C++ type casts are
     *  checked by the compiler, whereas C-style casts are not, and
     *  can fail at runtime.
     *
     *  Learn more about C++ type casting here:
     *  https://cplusplus.com/doc/oldtutorial/typecasting/
     *
     *  Refer to the parameter header for the maximum and minimum
     *  motor PWM values.
     *
     *  TODO LAB 7 YOUR CODE HERE.
     */
}

void
Controller::updateActiveStatus(const IMUData& imu_data)
{
    /*
     *  Check Y attitude (pitch)
     */
    if (imu_data.acceleration_z < 0
            || fabs(radiansToDegrees(imu_data.attitude_y))
                    > controller_parameter_.attitude_y_active)
    {
        /*
         *  The controller becomes inactive if the magnitude of the
         *  Y attitude (pitch) is greater than the Y attitude threshold
         *  in the parameter header.
         */
        active_ = false;

        /*
         *  Set NeoPixel frame to the class member inactive frame.
         */
        if (neopixel_)
        {
            neopixel_->setFrame(neopixel_frame_inactive_);
        }
    }
    else
    {
        /*
         *  If the controller was inactive and is now becoming active,
         *  reset all integrated errors.
         */
        if (!active_)
        {
            pid_controller_position_x_.resetErrorIntegral();
            pid_controller_attitude_y_.resetErrorIntegral();
            pid_controller_attitude_z_.resetErrorIntegral();
        }

        /*
         *  The controller is active if the magnitude of the
         *  Y attitude (pitch) is within the threshold.
         */
        active_ = true;

        /*
         *  Set NeoPixel frame to the class member active frame.
         */
        if (neopixel_)
        {
            neopixel_->setFrame(neopixel_frame_active_);
        }
    }
}
}   // namespace firmware
}   // namespace biped
