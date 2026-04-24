/**
 *  @file   maneuver_planner.cpp
 *  @author Simon Yu
 *  @date   04/01/2022
 *  @brief  Maneuver planner class source.
 *
 *  This file implements the maneuver planner class.
 */

/*
 *  Project headers.
 */
#include "common/global.h"
#include "common/type.h"
#include "controller/controller.h"
#include "planner/maneuver_planner.h"
#include "platform/serial.h"
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
ManeuverPlanner::ManeuverPlanner() : maneuver_counter_(1), maneuver_timer_(0), plan_started_(false),
        maneuver_started_(false), plan_completed_(true)
{
    /*
     *  Create a set of maneuvers for the example plan.
     *  In the following configurations, the maneuvers should
     *  be chained up in a linked list fashion.
     */
    last_gesture_recognized = 0;
    maneuver_thumbsup_ = std::make_shared<Maneuver>();
    maneuver_peace_ = std::make_shared<Maneuver>();
    maneuver_leftpointing_ = std::make_shared<Maneuver>();
    maneuver_rightpointing_ = std::make_shared<Maneuver>();
    park_ = std::make_shared<Maneuver>();

    maneuver_thumbsup_->transition_type = Maneuver::TransitionType::gesture_changed;
    maneuver_thumbsup_->transition_value = 0;
    maneuver_thumbsup_->type = Maneuver::Type::drive;
    maneuver_thumbsup_->next = nullptr;

    maneuver_peace_->transition_type = Maneuver::TransitionType::gesture_changed;
    maneuver_peace_->transition_value = 0;
    maneuver_peace_->type = Maneuver::Type::reverse;
    maneuver_peace_->next = nullptr;

    maneuver_leftpointing_->transition_type = Maneuver::TransitionType::gesture_changed;
    maneuver_leftpointing_->transition_value = 0;
    maneuver_leftpointing_->type = Maneuver::Type::drive_left;
    maneuver_leftpointing_->next = nullptr;

    maneuver_rightpointing_->transition_type = Maneuver::TransitionType::gesture_changed;
    maneuver_rightpointing_->transition_value = 0;
    maneuver_rightpointing_->type = Maneuver::Type::drive_right;
    maneuver_rightpointing_->next = nullptr;

    park_->transition_type = Maneuver::TransitionType::gesture_changed;
    park_->transition_value = 0;
    park_->type = Maneuver::Type::park;
    park_->next = nullptr;

    std::shared_ptr<Maneuver> maneuver_1 = std::make_shared<Maneuver>();


    /*
     *  Set the start and current maneuvers.
     */
    maneuver_start_ = maneuver_1;
    maneuver_ = maneuver_start_;

    /*
     *  Example plan maneuver 1 configuration:
     *      - Park for 2 seconds.
     *      - Then, start maneuver 2.
     */
    maneuver_1->transition_type = Maneuver::TransitionType::duration;
    maneuver_1->transition_value = 2;
    maneuver_1->type = Maneuver::Type::park;
    maneuver_1->next = park_;

    /*
     *  Example plan maneuver 2 configuration:
     *      - Drive forward until the X position goes above 1 meter.
     *      - Then, start maneuver 3.
     */
    

    /*
     *  Example plan maneuver 3 configuration:
     *      - Park for 2 seconds.
     *      - Then, start maneuver 4.
     */

    /*
     *  Using the example plan above, create your own maneuver-based plan.
     *  Feel free to add or remove maneuvers. Feel free to also remove or
     *  comment out the example plan.
     *
     *  Remember to initialize the class member start and current maneuver shared
     *  pointers to the first maneuver in your own maneuver-based plan.
     *
     *  Refer to the type header for the maneuver type and the maneuver transition
     *  type enum classes or all the available maneuver types and maneuver
     *  transition types.
     *
     *  TODO LAB 9 YOUR CODE HERE.
     */

//    maneuver_1->transition_type = Maneuver::TransitionType::duration;
//    maneuver_1->transition_value = 2;
//    maneuver_1->type = Maneuver::Type::park;
//    maneuver_1->next = maneuver_2;
//
//    /*
//     *  Example plan maneuver 2 configuration:
//     *      - Drive forward until the X position goes above 1 meter.
//     *      - Then, start maneuver 3.
//     */
//    maneuver_2->transition_type = Maneuver::TransitionType::range_middle_below;
//    maneuver_2->transition_value = 0.6;
//    maneuver_2->type = Maneuver::Type::drive;
//    maneuver_2->next = maneuver_3;
//
//    maneuver_3->transition_type = Maneuver::TransitionType::duration;
//    maneuver_3->transition_value = 2;
//    maneuver_3->type = Maneuver::Type::park;
//    maneuver_3->next = maneuver_4;
//
//
//    /*
//     *  Example plan maneuver 3 configuration:
//     *      - Park for 2 seconds.
//     *      - Then, start maneuver 4.
//     */
//    maneuver_4->transition_type = Maneuver::TransitionType::duration;
//    maneuver_4->transition_value = 5;
//    maneuver_4->type = Maneuver::Type::drive_left;
//    maneuver_4->next = maneuver_5;
//
//    maneuver_5->transition_type = Maneuver::TransitionType::duration;
//    maneuver_5->transition_value = 2;
//    maneuver_5->type = Maneuver::Type::park;
//    maneuver_5->next = maneuver_6;
//    /*
//     *  Example plan maneuver 4 configuration:
//     *      - Drive right until the X position goes above 2 meters.
//     *      - Then, start maneuver 5.
//     */
//    maneuver_6->transition_type = Maneuver::TransitionType::duration;
//    maneuver_6->transition_value = 2;
//    maneuver_6->type = Maneuver::Type::drive_right;
//    maneuver_6->next = maneuver_7;
//
//    /*
//     *  Example plan maneuver 5 configuration:
//     *      - Park for 2 seconds.
//     *      - The end.
//     */
//    maneuver_7->transition_type = Maneuver::TransitionType::position_x_above;
//    maneuver_7->transition_value = 4;
//    maneuver_7->type = Maneuver::Type::drive;
//    maneuver_7->next = maneuver_8;
//
//    maneuver_8->transition_type = Maneuver::TransitionType::duration;
//    maneuver_8->transition_value = 2;
//    maneuver_8->type = Maneuver::Type::park;
//    maneuver_8->next = nullptr;
}

void IRAM_ATTR
ManeuverPlanner::start()
{
    /*
     *  If the plan is completed, reset the class member current maneuver
     *  shared pointer to the class member start maneuver shared pointer,
     *  reset the class member maneuver counter to 1, mark maneuver as not
     *  started, and mark the plan as not started and not completed.
     *
     *  TODO LAB 8 YOUR CODE HERE.
     */
    if (plan_completed_) {
        maneuver_ = maneuver_start_;
        maneuver_counter_ = 1;
        maneuver_started_ = 0;
        plan_started_ = 0;
        plan_completed_ = 0;
    }

}

int
ManeuverPlanner::plan()
{
    /*
     *  Validate sensor global object shared pointer.
     */
    if (!sensor_)
    {
        Serial(LogLevel::error) << "Sensor missing.";
        return -1;
    }

    /*
     *  Validate controller global object shared pointer.
     */
    if (!controller_)
    {
        Serial(LogLevel::error) << "Controller missing.";
        return -1;
    }

    /*
     *  Return -1 if the plan is completed or if the controller
     *  is not active (pause the plan during safety disengage).
     *
     *  TODO LAB 8 YOUR CODE HERE.
     */
    if (plan_completed_ || !controller_->getActiveStatus()) {
        return -1;
    }
    /*
     *  Detect plan completion.
     */
    if (plan_started_ && !plan_completed_ && !maneuver_)
    {
        Serial(LogLevel::info) << "Completed maneuver-based plan.";

        /*
         *  Mark the plan as not started but completed and return
         *  -1 if the plan has started, has not completed, but the
         *  class member current maneuver shared pointer is a null
         *  pointer.
         *
         *  TODO LAB 8 YOUR CODE HERE.
         */
        plan_started_ = 0;
        plan_completed_ = 1;
        return -1;
    }

    if (!plan_started_)
    {
        /*
         *  The plan is empty if the plan has not started but the
         *  class member current maneuver shared pointer is already
         *  a null pointer. Return -1.
         */
        if (!maneuver_)
        {
            Serial(LogLevel::error) << "Empty maneuver-based plan.";
            return -1;
        }

        Serial(LogLevel::info) << "Started maneuver-based plan.";

        /*
         *  Otherwise, mark the plan as started if it has not started.
         */
        plan_started_ = true;
    }

    if (!maneuver_started_)
    {
        Serial(LogLevel::info) << "Started maneuver " << maneuver_counter_ << ".";

        /*
         *  Start and execute the current maneuver if it has not started.
         *
         *  Generate controller reference from the current maneuver and
         *  set the generated controller reference to the controller.
         *  Using the Arduino millis timing function, update the class member
         *  maneuver timer to the current time in milliseconds, and mark the
         *  current maneuver as started.
         *
         *  Refer to the controller header for the controller functions.
         *
         *  TODO LAB 8 YOUR CODE HERE.
         */
        maneuver_timer_ = millis();
        controller_->setControllerReference(generateControllerReference());
        maneuver_started_ = 1;
    }
    else
    {
        /*
         *  Transition to the next maneuver based on the current maneuver
         *  transition type and value.
         */
        switch (maneuver_->transition_type)
        {
            case Maneuver::TransitionType::duration:
            {
                /*
                 *  If the elapsed duration, i.e., the difference between the current
                 *  time in milliseconds given by the Arduino millis timing function and
                 *  the class member maneuver timer, goes above the transition value in
                 *  the current maneuver, transition to the next maneuver by setting the
                 *  class member current maneuver shared pointer to the next maneuver
                 *  shared pointer in the current maneuver, increment the class member
                 *  maneuver counter, and mark the current maneuver as not started.
                 *
                 *  Remember to convert the transition value to the correct unit using
                 *  the unit conversion functions in the math header.
                 *
                 *  Refer to the type header for the Maneuver struct entries.
                 *
                 *  TODO LAB 8 YOUR CODE HERE.
                 */
//printf("%.2f", secondsToMilliseconds(maneuver_->transition_value));
                if (millis() - maneuver_timer_ > secondsToMilliseconds(maneuver_->transition_value)) {
                    maneuver_ = maneuver_->next;
                    maneuver_counter_++;
                    maneuver_started_ = 0;
                }

                break;
            }
            case Maneuver::TransitionType::position_x_above:
            {
                /*
                 *  Obtain the encoder data using the sensor global object shared pointer.
                 *  If the current X position in the obtained encoder data goes above the
                 *  transition value in the current maneuver, transition to the next maneuver
                 *  by setting the class member current maneuver shared pointer to the next
                 *  maneuver shared pointer in the current maneuver, increment the class
                 *  member maneuver counter, and mark the current maneuver as not started.
                 *
                 *  Refer to the type header for the Maneuver struct entries and the sensor
                 *  header for the sensor functions.
                 *
                 *  TODO LAB 8 YOUR CODE HERE.
                 */
                double cur_x = sensor_->getEncoderData().position_x;
                if (cur_x > maneuver_->transition_value) {
                    maneuver_ = maneuver_->next;
                    maneuver_counter_++;
                    maneuver_started_ = 0;
                }

                break;
            }
            case Maneuver::TransitionType::position_x_below:
            {
                /*
                 *  Obtain the encoder data using the sensor global object shared pointer.
                 *  If the current X position in the obtained encoder data goes below the
                 *  transition value in the current maneuver, transition to the next maneuver
                 *  by setting the class member current maneuver shared pointer to the next
                 *  maneuver shared pointer in the current maneuver, increment the class
                 *  member maneuver counter, and mark the current maneuver as not started.
                 *
                 *  Refer to the type header for the Maneuver struct entries and the sensor
                 *  header for the sensor functions.
                 *
                 *  TODO LAB 8 YOUR CODE HERE.
                 */
                double cur_x = sensor_->getEncoderData().position_x;
                if (cur_x < maneuver_->transition_value) {
                    maneuver_ = maneuver_->next;
                    maneuver_counter_++;
                    maneuver_started_ = 0;
                }


                break;
            }
            case Maneuver::TransitionType::range_left_above:
            {
                /*
                 *  Obtain the time-of-flight data using the sensor global object shared pointer.
                 *  If the left time-of-flight range in the obtained time-of-flight data goes
                 *  above the transition value in the current maneuver, transition to the next
                 *  maneuver by setting the class member current maneuver shared pointer to the
                 *  next maneuver shared pointer in the current maneuver, increment the class
                 *  member maneuver counter, and mark the current maneuver as not started.
                 *
                 *  Refer to the type header for the Maneuver struct entries and the sensor
                 *  header for the sensor functions.
                 *
                 *  TODO LAB 8 YOUR CODE HERE.
                 */
                double cur_left = sensor_->getTimeOfFlightData().range_right;
                if (cur_left > maneuver_->transition_value) {
                    maneuver_ = maneuver_->next;
                    maneuver_counter_++;
                    maneuver_started_ = 0;
                }

                break;
            }
            case Maneuver::TransitionType::range_left_below:
            {
                /*
                 *  Obtain the time-of-flight data using the sensor global object shared pointer.
                 *  If the left time-of-flight range in the obtained time-of-flight data goes
                 *  below the transition value in the current maneuver, transition to the next
                 *  maneuver by setting the class member current maneuver shared pointer to the
                 *  next maneuver shared pointer in the current maneuver, increment the class
                 *  member maneuver counter, and mark the current maneuver as not started.
                 *
                 *  Refer to the type header for the Maneuver struct entries and the sensor
                 *  header for the sensor functions.
                 *
                 *  TODO LAB 8 YOUR CODE HERE.
                 */
                double cur_left = sensor_->getTimeOfFlightData().range_right;
                if (cur_left < maneuver_->transition_value) {
                    maneuver_ = maneuver_->next;
                    maneuver_counter_++;
                    maneuver_started_ = 0;
                }
                break;
            }
            case Maneuver::TransitionType::range_middle_above:
            {
                /*
                 *  Obtain the time-of-flight data using the sensor global object shared pointer.
                 *  If the middle time-of-flight range in the obtained time-of-flight data goes
                 *  above the transition value in the current maneuver, transition to the next
                 *  maneuver by setting the class member current maneuver shared pointer to the
                 *  next maneuver shared pointer in the current maneuver, increment the class
                 *  member maneuver counter, and mark the current maneuver as not started.
                 *
                 *  Refer to the type header for the Maneuver struct entries and the sensor
                 *  header for the sensor functions.
                 *
                 *  TODO LAB 8 YOUR CODE HERE.
                 */
                double cur_mid = sensor_->getTimeOfFlightData().range_middle;
                if (cur_mid > maneuver_->transition_value) {
                    maneuver_ = maneuver_->next;
                    maneuver_counter_++;
                    maneuver_started_ = 0;
                }

                break;
            }
            case Maneuver::TransitionType::range_middle_below:
            {
                /*
                 *  Obtain the time-of-flight data using the sensor global object shared pointer.
                 *  If the middle time-of-flight range in the obtained time-of-flight data goes
                 *  below the transition value in the current maneuver, transition to the next
                 *  maneuver by setting the class member current maneuver shared pointer to the
                 *  next maneuver shared pointer in the current maneuver, increment the class
                 *  member maneuver counter, and mark the current maneuver as not started.
                 *
                 *  Refer to the type header for the Maneuver struct entries and the sensor
                 *  header for the sensor functions.
                 *
                 *  TODO LAB 8 YOUR CODE HERE.
                 */
                double cur_mid = sensor_->getTimeOfFlightData().range_middle;
                if (cur_mid < maneuver_->transition_value) {
                    maneuver_ = maneuver_->next;
                    maneuver_counter_++;
                    maneuver_started_ = 0;
                }

                break;
            }
            case Maneuver::TransitionType::range_right_above:
            {
                /*
                 *  Obtain the time-of-flight data using the sensor global object shared pointer.
                 *  If the right time-of-flight range in the obtained time-of-flight data goes
                 *  above the transition value in the current maneuver, transition to the next
                 *  maneuver by setting the class member current maneuver shared pointer to the
                 *  next maneuver shared pointer in the current maneuver, increment the class
                 *  member maneuver counter, and mark the current maneuver as not started.
                 *
                 *  Refer to the type header for the Maneuver struct entries and the sensor
                 *  header for the sensor functions.
                 *
                 *  TODO LAB 8 YOUR CODE HERE.
                 */
                double cur_right = sensor_->getTimeOfFlightData().range_right;
                if (cur_right > maneuver_->transition_value) {
                    maneuver_ = maneuver_->next;
                    maneuver_counter_++;
                    maneuver_started_ = 0;
                }

                break;
            }
            case Maneuver::TransitionType::range_right_below:
            {
                /*
                 *  Obtain the time-of-flight data using the sensor global object shared pointer.
                 *  If the right time-of-flight range in the obtained time-of-flight data goes
                 *  above the transition value in the current maneuver, transition to the next
                 *  maneuver by setting the class member current maneuver shared pointer to the
                 *  next maneuver shared pointer in the current maneuver, increment the class
                 *  member maneuver counter, and mark the current maneuver as not started.
                 *
                 *  Refer to the type header for the Maneuver struct entries and the sensor
                 *  header for the sensor functions.
                 *
                 *  TODO LAB 8 YOUR CODE HERE.
                 */
                double cur_right = sensor_->getTimeOfFlightData().range_right;
                if (cur_right < maneuver_->transition_value) {
                    maneuver_ = maneuver_->next;
                    maneuver_counter_++;
                    maneuver_started_ = 0;
                }


                break;
            }
            case Maneuver::TransitionType::gesture_changed:
            {
                //printf("%u\n", gesture_recognized_);
                if (gesture_recognized_ != last_gesture_recognized) {
                    switch (gesture_recognized_) {
                        case 1:
                        {
                            maneuver_ = maneuver_thumbsup_;
                            maneuver_counter_++;
                            maneuver_started_ = 0;
                            break;
                        }
                        case 2:
                        {
                            maneuver_ = maneuver_peace_;
                            maneuver_counter_++;
                            maneuver_started_ = 0;
                            break;
                        }
                        case 3:
                        {
                            maneuver_ = maneuver_leftpointing_;
                            maneuver_counter_++;
                            maneuver_started_ = 0;
                            break;
                        }
                        case 4:
                        {
                            maneuver_ = maneuver_rightpointing_;
                            maneuver_counter_++;
                            maneuver_started_ = 0;
                            break;
                        }
                        default:
                        {
                            maneuver_ = park_;
                            maneuver_counter_++;
                            maneuver_started_ = 0;
                            break;
                        }
                    }
                }
                last_gesture_recognized = gesture_recognized_;
                break;
            }
            default:
            {
                /*
                 *  Print a warning message to serial for unknown maneuver transition type.
                 */
                Serial(LogLevel::warn) << "Unknown maneuver transition type.";
                break;
            }
        }
    }

    /*
     *  Return the class member maneuver counter.
     *
     *  TODO LAB 8 YOUR CODE HERE.
     */
    return maneuver_counter_;
}

ControllerReference
ManeuverPlanner::generateControllerReference() const
{
    /*
     *  Declare a default controller reference struct.
     */
    ControllerReference controller_reference;

    /*
     *  Validate class member current maneuver shared pointer.
     */
    if (!maneuver_)
    {
        Serial(LogLevel::warn) << "Invalid maneuver.";
        return controller_reference;
    }

    /*
     *  Generate the controller references based on the current
     *  maneuver type.
     */
    switch (maneuver_->type)
    {
        case Maneuver::Type::park:
        {
            /*
             *  Obtain the encoder data using the sensor global object shared pointer.
             *  Set the X position controller reference to be the current X position
             *  in the obtained encoder data, i.e., remain at the current X position.
             *
             *  TODO LAB 8 YOUR CODE HERE.
             */
            controller_reference = controller_->getControllerReference();
            controller_reference.position_x = sensor_->getEncoderData().position_x;
            controller_->setControllerReference(controller_reference);

            break;
        }
        case Maneuver::Type::reverse:
        {
            /*
             *  Obtain the encoder data using the sensor global object shared pointer.
             *  Set the X position controller reference to be 1000 meters below the
             *  current X position in the obtained encoder data, i.e., reverse indefinitely
             *  until the next maneuver.
             *
             *  TODO LAB 8 YOUR CODE HERE.
             */
            controller_reference = controller_->getControllerReference();
            controller_reference.position_x = sensor_->getEncoderData().position_x - 1000;
            controller_->setControllerReference(controller_reference);

            break;
        }
        case Maneuver::Type::reverse_left:
        {
            /*
             *  Obtain the encoder data using the sensor global object shared pointer.
             *  Set the X position controller reference to be 1000 meters below the
             *  current X position in the obtained encoder data, i.e., reverse indefinitely
             *  until the next maneuver. Set the Z attitude controller reference to reverse
             *  left 90 degrees.
             *
             *  Note that for the reverse turning directions, treat the reversing Biped as
             *  if it is driving forward. Then, the direction the Biped is turning in reverse
             *  is its reverse direction.
             *
             *  Remember to convert all angles to radians using the conversion functions
             *  in the math header before assigning them to the controller reference struct.
             *
             *  TODO LAB 8 YOUR CODE HERE.
             */
            controller_reference = controller_->getControllerReference();
            controller_reference.position_x = sensor_->getEncoderData().position_x - 1000;
            controller_reference.attitude_z -= degreesToRadians(90); //unsure
            controller_->setControllerReference(controller_reference);

            break;
        }
        case Maneuver::Type::reverse_right:
        {
            /*
             *  Obtain the encoder data using the sensor global object shared pointer.
             *  Set the X position controller reference to be 1000 meters below the
             *  current X position in the obtained encoder data, i.e., reverse indefinitely
             *  until the next maneuver. Set the Z attitude controller reference to reverse
             *  right 90 degrees.
             *
             *  Note that for the reverse turning directions, treat the reversing Biped as
             *  if it is driving forward. Then, the direction the Biped is turning in reverse
             *  is its reverse direction.
             *
             *  Remember to convert all angles to radians using the conversion functions
             *  in the math header before assigning them to the controller reference struct.
             *
             *  TODO LAB 8 YOUR CODE HERE.
             */
            controller_reference = controller_->getControllerReference();
            controller_reference.position_x = sensor_->getEncoderData().position_x - 1000;
            controller_reference.attitude_z += degreesToRadians(90); //unsure
            controller_->setControllerReference(controller_reference);

            break;
        }
        case Maneuver::Type::drive:
        {
            /*
             *  Obtain the encoder data using the sensor global object shared pointer.
             *  Set the X position controller reference to be 1000 meters above the
             *  current X position in the obtained encoder data, i.e., drive indefinitely
             *  until the next maneuver.
             *
             *  TODO LAB 8 YOUR CODE HERE.
             */
            controller_reference = controller_->getControllerReference();
            controller_reference.position_x = sensor_->getEncoderData().position_x + 1000;
            controller_->setControllerReference(controller_reference);

            break;
        }
        case Maneuver::Type::drive_left:
        {
            /*
             *  Obtain the encoder data using the sensor global object shared pointer.
             *  Set the X position controller reference to be 1000 meters above the
             *  current X position in the obtained encoder data, i.e., drive indefinitely
             *  until the next maneuver. Set the Z attitude controller reference to drive
             *  left 90 degrees.
             *
             *  Remember to convert all angles to radians using the conversion functions
             *  in the math header before assigning them to the controller reference struct.
             *
             *  TODO LAB 8 YOUR CODE HERE.
             */
            controller_reference = controller_->getControllerReference();
            controller_reference.position_x = sensor_->getEncoderData().position_x + 1000;
            controller_reference.attitude_z = degreesToRadians(-180);
            controller_->setControllerReference(controller_reference);


            break;
        }
        case Maneuver::Type::drive_right:
        {
            /*
             *  Obtain the encoder data using the sensor global object shared pointer.
             *  Set the X position controller reference to be 1000 meters above the
             *  current X position in the obtained encoder data, i.e., drive indefinitely
             *  until the next maneuver. Set the Z attitude controller reference to drive
             *  right 90 degrees.
             *
             *  Remember to convert all angles to radians using the conversion functions
             *  in the math header before assigning them to the controller reference struct.
             *
             *  TODO LAB 8 YOUR CODE HERE.
             */
            controller_reference = controller_->getControllerReference();
            controller_reference.position_x = sensor_->getEncoderData().position_x + 1000;
            controller_reference.attitude_z = degreesToRadians(180);
            controller_->setControllerReference(controller_reference);
            break;
        }
        default:
        {
            /*
             *  Print a warning message to serial for unknown maneuver type.
             */
            Serial(LogLevel::warn) << "Unknown maneuver type.";
            break;
        }
    }

    /*
     *  Return the updated controller reference struct.
     */
    return controller_reference;
}
}   // namespace firmware
}   // namespace biped
