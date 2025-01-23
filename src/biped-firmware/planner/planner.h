/**
 *  @file   planner.h
 *  @author Simon Yu
 *  @date   04/01/2022
 *  @brief  Planner abstract class header.
 *
 *  This file defines the planner abstract class.
 */

/*
 *  Include guard.
 */
#ifndef PLANNER_PLANNER_H_
#define PLANNER_PLANNER_H_

/*
 *  External headers.
 */
#include <esp_attr.h>

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
 *  @brief  Planner abstract class.
 *
 *  This abstract class provides pure virtual functions
 *  for various types of child planner classes to inherit
 *  and implement.
 *
 *  Learn more about C++ Polymorphism here:
 *  https://www.w3schools.com/cpp/cpp_polymorphism.asp
 */
class Planner
{
public:

    /**
     *  @brief  Planner abstract class pure virtual destructor.
     *
     *  This destructor is pure virtual and prohibits the
     *  instantiation of this abstract class.
     */
    virtual
    ~Planner() = default;

    /**
     *  @brief  Start plan.
     *
     *  This function is pure virtual, and its functionality
     *  is to be implemented by any child planner class.
     */
    virtual void IRAM_ATTR
    start() = 0;

    /**
     *  @return Planner stage.
     *  @brief  Execute plan.
     *
     *  This function is pure virtual, and its functionality
     *  is to be implemented by any child planner class.
     */
    virtual int
    plan() = 0;
};
}   // namespace firmware
}   // namespace biped

#endif  // PLANNER_PLANNER_H_
