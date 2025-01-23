/**
 *  @file   type.h
 *  @author Simon Yu
 *  @date   01/05/2023
 *  @brief  Type class header.
 *
 *  This file defines the type classes.
 */

/*
 *  Include guard.
 */
#ifndef COMMON_TYPE_H_
#define COMMON_TYPE_H_

/*
 *  External headers.
 */
#include <limits>
#include <memory>

/*
 *  Project headers.
 */
#include "utility/serializer.hpp"

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
 *  @brief  Actuation command struct.
 *
 *  This struct contains actuation command entries,
 *  such as the motor enable, left and right motor direction
 *  and pulse width modulation (PWM) values.
 */
struct ActuationCommand
{
    /*
     *  This struct is a friend of the zpp serializer
     *  access class.
     */
    friend zpp::serializer::access;

    bool motor_enable;  //!< Motor enable.
    bool motor_left_forward;    //!< Left motor direction.
    bool motor_right_forward;   //!< Right motor direction.
    double motor_left_pwm;   //!< Left motor PWM.
    double motor_right_pwm;  //!< Right motor PWM.

    /**
     *  @brief  Actuation command struct constructor.
     *
     *  This constructor initializes all actuation command struct entries.
     */
    ActuationCommand() : motor_enable(false), motor_left_forward(true), motor_right_forward(false),
            motor_left_pwm(0), motor_right_pwm(0)
    {
    }

    /**
     *  @tparam Archive Type of archive.
     *  @tparam Self Type of this struct.
     *  @param  archive Serialization archive.
     *  @param  self This struct.
     *  @return Serialization archive.
     *  @brief  Actuation command serialization function.
     *
     *  This function performs serialization of all entries in this struct.
     */
    template<typename Archive, typename Self>
    inline static auto
    serialize(Archive& archive, Self& self)
    {
        return archive(self.motor_enable, self.motor_left_forward, self.motor_right_forward,
                self.motor_left_pwm, self.motor_right_pwm);
    }
};

/**
 *  @brief  Controller reference struct.
 *
 *  This struct contains controller reference entries,
 *  such as X position (forward/backward), Y attitude (pitch), and
 *  Z attitude (yaw) references.
 */
struct ControllerReference
{
    /*
     *  This struct is a friend of the zpp serializer
     *  access class.
     */
    friend zpp::serializer::access;

    double attitude_y;   //!< Y attitude (pitch) controller reference, in radians.
    double attitude_z;   //!< Z attitude (yaw) controller reference, in radians.
    double position_x;   //!< X position controller reference, in meters.

    /**
     *  @brief  Controller reference struct constructor.
     *
     *  This constructor initializes all controller reference struct entries to 0.
     */
    ControllerReference() : attitude_y(0), attitude_z(0), position_x(0)
    {
    }

    /**
     *  @tparam Archive Type of archive.
     *  @tparam Self Type of this struct.
     *  @param  archive Serialization archive.
     *  @param  self This struct.
     *  @return Serialization archive.
     *  @brief  Controller reference serialization function.
     *
     *  This function performs serialization of all entries in this struct.
     */
    template<typename Archive, typename Self>
    inline static auto
    serialize(Archive& archive, Self& self)
    {
        return archive(self.attitude_y, self.attitude_z, self.position_x);
    }
};

/**
 *  @brief  Controller saturation struct.
 *
 *  This struct contains controller saturation entries,
 *  such as the input and output saturations.
 *
 *  Learn more about controller saturation here:
 *  https://www.dmi.unict.it/santoro/teaching/sr/slides/PIDSaturation.pdf
 */
struct ControllerSaturation
{
    /*
     *  This struct is a friend of the zpp serializer
     *  access class.
     */
    friend zpp::serializer::access;

    double input_upper;   //!< Input saturation upper bound.
    double input_lower;   //!< Input saturation lower bound.
    double output_upper;  //!< Output saturation upper bound.
    double output_lower;  //!< Output saturation lower bound.

    /**
     *  @brief  Controller saturation struct constructor.
     *
     *  This constructor initializes all controller saturation
     *  entries to their respective extremes.
     */
    ControllerSaturation() : input_upper(std::numeric_limits<double>::max()),
            input_lower(std::numeric_limits<double>::lowest()),
            output_upper(std::numeric_limits<double>::max()),
            output_lower(std::numeric_limits<double>::lowest())
    {
    }

    /**
     *  @tparam Archive Type of archive.
     *  @tparam Self Type of this struct.
     *  @param  archive Serialization archive.
     *  @param  self This struct.
     *  @return Serialization archive.
     *  @brief  Controller saturation serialization function.
     *
     *  This function performs serialization of all entries in this struct.
     */
    template<typename Archive, typename Self>
    inline static auto
    serialize(Archive& archive, Self& self)
    {
        return archive(self.input_upper, self.input_lower, self.output_upper, self.output_lower);
    }
};

/**
 *  @brief  Encoder data struct.
 *
 *  This struct contains encoder data entries.
 *
 *  All spatial data is in the standard body reference frame.
 *
 *  Refer to the definition of the standard body reference frame here:
 *  https://www.vectornav.com/resources/inertial-navigation-primer/math-fundamentals/math-refframes
 */
struct EncoderData
{
    /*
     *  This struct is a friend of the zpp serializer
     *  access class.
     */
    friend zpp::serializer::access;

    double position_x;   //!< X position, in meters.
    double steps;   //!< Overall encoder steps.
    double steps_left;  //!< Left encoder steps.
    double steps_right; //!< Right encoder steps.
    double velocity_x;   //!< X velocity, in meters per second.

    /**
     *  @brief  Encoder data struct constructor.
     *
     *  This constructor initializes all encoder data struct entries to 0.
     */
    EncoderData() : position_x(0), steps(0), steps_left(0), steps_right(0), velocity_x(0)
    {
    }

    /**
     *  @tparam Archive Type of archive.
     *  @tparam Self Type of this struct.
     *  @param  archive Serialization archive.
     *  @param  self This struct.
     *  @return Serialization archive.
     *  @brief  Encoder data serialization function.
     *
     *  This function performs serialization of all entries in this struct.
     */
    template<typename Archive, typename Self>
    inline static auto
    serialize(Archive& archive, Self& self)
    {
        return archive(self.position_x, self.steps, self.steps_left, self.steps_right,
                self.velocity_x);
    }
};

/**
 *  @brief  IMU data struct.
 *
 *  This struct contains inertial measurement unit (IMU) data entries,
 *  such as attitude, linear velocity, angular velocity,
 *  linear acceleration, and etc.
 *
 *  All spatial data is in the standard body reference frame.
 *
 *  Refer to the definition of the standard body reference frame here:
 *  https://www.vectornav.com/resources/inertial-navigation-primer/math-fundamentals/math-refframes
 *
 *  Refer to the rotational right-hand rule here:
 *  https://en.wikipedia.org/wiki/Right-hand_rule#Rotations
 */
struct IMUData
{
    /*
     *  This struct is a friend of the zpp serializer
     *  access class.
     */
    friend zpp::serializer::access;

    double acceleration_x;   //!< X acceleration, in meters per second squared.
    double acceleration_y;   //!< Y acceleration, in meters per second squared.
    double acceleration_z;   //!< Z acceleration, in meters per second squared.
    double attitude_x;   //!< X attitude, or roll angle, in radians.
    double attitude_y;   //!< Y attitude, or pitch angle, in radians.
    double attitude_z;   //!< Z attitude, or yaw angle, in radians.
    double angular_velocity_x;   //!< X angular velocity, or roll rate, in radians per second.
    double angular_velocity_y;   //!< Y angular velocity, or pitch rate, in radians per second.
    double angular_velocity_z;   //!< Z angular velocity, or yaw rate, in radians per second.
    double compass_x;   //!< X compass field strength, in micro Tesla.
    double compass_y;   ///!< Y compass field strength, In micro Tesla.
    double compass_z;   //!< Z compass field strength, In micro Tesla.
    double temperature;  //!< Temperature, in Celsius.

    /**
     *  @brief  IMU data struct constructor.
     *
     *  This constructor initializes all IMU data struct entries to 0.
     */
    IMUData() : acceleration_x(0), acceleration_y(0), acceleration_z(0), attitude_x(0),
            attitude_y(0), attitude_z(0), angular_velocity_x(0), angular_velocity_y(0),
            angular_velocity_z(0), compass_x(0), compass_y(0), compass_z(0), temperature(0)
    {
    }

    /**
     *  @tparam Archive Type of archive.
     *  @tparam Self Type of this struct.
     *  @param  archive Serialization archive.
     *  @param  self This struct.
     *  @return Serialization archive.
     *  @brief  IMU data serialization function.
     *
     *  This function performs serialization of all entries in this struct.
     */
    template<typename Archive, typename Self>
    inline static auto
    serialize(Archive& archive, Self& self)
    {
        return archive(self.acceleration_x, self.acceleration_y, self.acceleration_z,
                self.attitude_x, self.attitude_y, self.attitude_z, self.angular_velocity_x,
                self.angular_velocity_y, self.angular_velocity_z, self.compass_x, self.compass_y,
                self.compass_z, self.temperature);
    }
};

/**
 *  @brief  Log level enum class.
 *
 *  This enum class defines log levels.
 */
enum class LogLevel
{
    fatal = 0,  //!< Fatal log level.
    error,  //!< Error log level.
    warn,   //!< Warning log level.
    info,   //!< Information log level.
    debug,  //!< Debugging log level.
    trace   //!< Tracing log level.
};

/**
 *  @brief  Planner maneuver struct
 *
 *  This struct contains planner maneuver entries,
 *  such as maneuver transition type, maneuver transition value,
 *  maneuver type, and the shared pointer to the next maneuver.
 *  The maneuvers can be chained up in a linked list fashion.
 *  The struct also defines the planner maneuver type and
 *  maneuver transition type enum classes.
 */
struct Maneuver
{
    /**
     *  @brief  Planner maneuver type enum class
     *
     *  This enum class defines maneuver types.
     */
    enum class Type
    {
        drive = 0,  //!< Drive forward until the end of this maneuver.
        drive_left, //!< Drive to the left until the end of this maneuver.
        drive_right,    //!< Drive to the right until the end of this maneuver.
        park,   //!< Park or stop until the end of this maneuver.
        reverse,    //!< Reverse until the end of this maneuver.
        reverse_left,   //!< Reverse to the left until the end of this maneuver.
        reverse_right,  //!< Reverse to the right until the end of this maneuver.
    };

    /**
     *  @brief  Planner maneuver transition type enum class
     *
     *  This enum class defines maneuver transition types.
     */
    enum class TransitionType
    {
        duration, //!< Transition after a certain time duration, in seconds.
        position_x_above, //!< Transition if the X position is above a certain value, in meters.
        position_x_below, //!< Transition if the X position is below a certain value, in meters.
        range_left_above, //!< Transition if the left time-of-flight range is above a certain value, in meters.
        range_left_below, //!< Transition if the left time-of-flight range is below a certain value, in meters.
        range_middle_above, //!< Transition if the middle time-of-flight range is above a certain value, in meters.
        range_middle_below, //!< Transition if the middle time-of-flight range is below a certain value, in meters.
        range_right_above, //!< Transition if the right time-of-flight range is above a certain value, in meters.
        range_right_below //!< Transition if the right time-of-flight range is below a certain value, in meters.
    };

    TransitionType transition_type; //!< Maneuver transition type.
    double transition_value; //!< Maneuver transition value, the meaning of which depends on the maneuver transition type.
    Type type;  //!< Maneuver type.
    std::shared_ptr<Maneuver> next; //!< Shared pointer to the next maneuver.

    /**
     *  @brief  Planner maneuver struct constructor
     *
     *  This constructor initializes all planner maneuver struct entries.
     */
    Maneuver() : transition_type(TransitionType::duration), transition_value(0), type(Type::park),
            next(nullptr)
    {
    }
};

/**
 *  @brief  PID controller gain struct.
 *
 *  This struct contains PID controller gain entries,
 *  such as the proportional, integral, differential
 *  gains, and etc.
 *
 *  Refer to Lecture 13 for the variable definitions.
 */
struct PIDControllerGain
{
    /*
     *  This struct is a friend of the zpp serializer
     *  access class.
     */
    friend zpp::serializer::access;

    double differential; //!< Differential gain (Kd).
    double integral; //!< Integral gain (Ki).
    double integral_max; //!< Maximum integrated error.
    double proportional; //!< Proportional gain (Kp).

    /**
     *  @brief  PID controller gain struct constructor.
     *
     *  This constructor initializes all PID controller gain entries to 0.
     */
    PIDControllerGain() : differential(0), integral(0), integral_max(0), proportional(0)
    {
    }

    /**
     *  @tparam Archive Type of archive.
     *  @tparam Self Type of this struct.
     *  @param  archive Serialization archive.
     *  @param  self This struct.
     *  @return Serialization archive.
     *  @brief  PID controller serialization function.
     *
     *  This function performs serialization of all entries in this struct.
     */
    template<typename Archive, typename Self>
    inline static auto
    serialize(Archive& archive, Self& self)
    {
        return archive(self.differential, self.integral, self.integral_max, self.proportional);
    }
};

/**
 *  @brief  Controller parameter struct.
 *
 *  This struct contains controller parameter entries,
 *  such as proportional, differential, and integral gains.
 */
struct ControllerParameter
{
    /*
     *  This struct is a friend of the zpp serializer
     *  access class.
     */
    friend zpp::serializer::access;

    double attitude_y_active; //!< Maximum Y attitude (pitch) for the controller to remain active, in degrees.
    double attitude_z_gain_open_loop;    //!< Z attitude open-loop controller gain.
    ControllerSaturation open_loop_controller_saturation_attitude_z; //!< Z attitude (yaw) open-loop controller saturation struct.
    PIDControllerGain pid_controller_gain_attitude_y; //!< Y attitude (pitch) PID controller gain struct.
    PIDControllerGain pid_controller_gain_attitude_z; //!< Z attitude (yaw) PID controller gain struct.
    PIDControllerGain pid_controller_gain_position_x; //!< X position (forward/backward) PID controller gain struct.
    ControllerSaturation pid_controller_saturation_attitude_y; //!< Y attitude (pitch) PID controller saturation struct.
    ControllerSaturation pid_controller_saturation_attitude_z; //!< Z attitude (yaw) PID controller saturation struct.
    ControllerSaturation pid_controller_saturation_position_x; //!< X position (forward/backward) PID controller saturation struct.

    /**
     *  @brief  Controller parameter struct constructor.
     *
     *  This constructor initializes controller parameter struct entries.
     */
    ControllerParameter() : attitude_y_active(20), attitude_z_gain_open_loop(0)
    {
    }

    /**
     *  @tparam Archive Type of archive.
     *  @tparam Self Type of this struct.
     *  @param  archive Serialization archive.
     *  @param  self This struct.
     *  @return Serialization archive.
     *  @brief  Controller parameter serialization function.
     *
     *  This function performs serialization of all entries in this struct.
     */
    template<typename Archive, typename Self>
    inline static auto
    serialize(Archive& archive, Self& self)
    {
        return archive(self.attitude_y_active, self.attitude_z_gain_open_loop,
                self.open_loop_controller_saturation_attitude_z,
                self.pid_controller_gain_attitude_y, self.pid_controller_gain_attitude_z,
                self.pid_controller_gain_position_x, self.pid_controller_saturation_attitude_y,
                self.pid_controller_saturation_attitude_z,
                self.pid_controller_saturation_position_x);
    }
};

/**
 *  @brief  Stream manipulator enum class.
 *
 *  This enum class defines stream manipulators.
 */
enum class StreamManipulator
{
    carriage_return,    //!< Carriage return stream manipulator.
    clear_line, //!< Clear-line stream manipulator.
    endl    //!< End-line stream manipulator.
};

/**
 *  @brief  Time-of-flight data struct.
 *
 *  This struct contains time-of-flight data entries.
 */
struct TimeOfFlightData
{
    /*
     *  This struct is a friend of the zpp serializer
     *  access class.
     */
    friend zpp::serializer::access;

    double range_left;   //!< Left time-of-flight range, in meters.
    double range_middle; //!< Middle time-of-flight range, in meters.
    double range_right;   //!< Right time-of-flight range, in meters.

    /**
     *  @brief  Time-of-flight data struct constructor
     *
     *  This constructor initializes all time-of-flight data struct entries.
     */
    TimeOfFlightData() : range_left(0), range_middle(0), range_right(0)
    {
    }

    /**
     *  @tparam Archive Type of archive.
     *  @tparam Self Type of this struct.
     *  @param  archive Serialization archive.
     *  @param  self This struct.
     *  @return Serialization archive.
     *  @brief  Time-of-flight serialization function.
     *
     *  This function performs serialization of all entries in this struct.
     */
    template<typename Archive, typename Self>
    inline static auto
    serialize(Archive& archive, Self& self)
    {
        return archive(self.range_left, self.range_middle, self.range_right);
    }
};

/**
 *  @brief  Planner waypoint struct
 *
 *  This struct contains planner waypoint entries,
 *  such as controller reference, waypoint duration,
 *  and the shared pointer to the next waypoint. The
 *  waypoints can be chained up in a linked list fashion.
 */
struct Waypoint
{
    ControllerReference controller_reference; //!< Controller reference.
    double duration; //!< Transition to the next waypoint after this time duration, in seconds.
    std::shared_ptr<Waypoint> next; //!< Shared pointer to the next waypoint.

    /**
     *  @brief  Planner waypoint struct constructor
     *
     *  This constructor initializes all planner waypoint struct entries.
     */
    Waypoint() : controller_reference(), duration(0), next(nullptr)
    {
    }
};

/**
 *  @brief  Biped message struct.
 *
 *  This struct contains various entries,
 *  such as the actuation command struct, controller structs,
 *  sensor data structs, and etc. This struct is typically used
 *  as a data communication protocol between the Biped firmware
 *  and the Biped ground station.
 */
struct BipedMessage
{
    /*
     *  This struct is a friend of the zpp serializer
     *  access class.
     */
    friend zpp::serializer::access;

    ActuationCommand actuation_command;    //!< Actuation command struct.
    ControllerParameter controller_parameter;  //!< Controller parameter struct.
    ControllerReference controller_reference;  //!< Controller reference struct.
    EncoderData encoder_data;  //!< Encoder data struct.
    IMUData imu_data;   //!< IMU data struct.
    unsigned long long sequence;   //!< Message sequence number.
    unsigned long long timestamp;   //!< Message timestamp, in microseconds.
    TimeOfFlightData time_of_flight_data; //!< Time-of-flight data struct.

    /**
     *  @brief  Biped message constructor
     *
     *  This constructor initializes all Biped message struct entries.
     */
    BipedMessage() : sequence(0), timestamp(0)
    {
    }

    /**
     *  @tparam Archive Type of archive.
     *  @tparam Self Type of this struct.
     *  @param  archive Serialization archive.
     *  @param  self This struct.
     *  @return Serialization archive.
     *  @brief  Biped message serialization function.
     *
     *  This function performs serialization of all entries in this struct.
     */
    template<typename Archive, typename Self>
    inline static auto
    serialize(Archive& archive, Self& self)
    {
        return archive(self.actuation_command, self.controller_parameter, self.controller_reference,
                self.encoder_data, self.imu_data, self.sequence, self.timestamp,
                self.time_of_flight_data);
    }
};
}   // namespace firmware
}   // namespace biped

#endif  // COMMON_TYPE_H_
