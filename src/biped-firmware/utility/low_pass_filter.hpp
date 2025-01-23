/**
 *  @file   low_pass_filter.hpp
 *  @author Simon Yu
 *  @date   01/19/2022
 *  @brief  Low-pass filter templated class header.
 *
 *  This file defines and implements the low-pass filter
 *  templated class.
 */

/*
 *  Include guard.
 */
#ifndef UTILITY_LOW_PASS_FILTER_H_
#define UTILITY_LOW_PASS_FILTER_H_

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
 *  @tparam Type Type of data for filtering.
 *  @brief  Low-pass filter templated class.
 *
 *  This templated class provides functions for
 *  filtering data using a simple, discrete
 *  low-pass filter.
 *
 *  Learn more about discrete low-pass filters here:
 *  https://en.wikipedia.org/wiki/Low-pass_filter#Difference_equation_through_discrete_time_sampling
 */
template<typename Type>
class LowPassFilter
{
public:

    /**
     *  @brief  Low-pass filter templated class constructor.
     *
     *  This constructor initializes all class member variables.
     */
    inline
    LowPassFilter() : data_filtered_(0), beta_(0)
    {
    }

    /**
     *  @param  beta Low-pass filter beta.
     *  @brief  Set the low-pass filter beta.
     *
     *  This function sets the low-pass filter beta.
     */
    inline void
    setBeta(const double& beta)
    {
        /*
         *  Set the class member low-pass filter beta.
         */
        beta_ = beta;
    }

    /**
     *  @tparam Type Type of filtered data.
     *  @param  data New data.
     *  @return Templated filtered data.
     *  @brief  Filter data using the discrete low-pass filter.
     *
     *  This function filters the new data using a simple, discrete
     *  low-pass filter implementation.
     */
    inline Type
    filter(const Type& data)
    {
        /*
         *  Learn more about discrete low-pass filters here:
         *  https://en.wikipedia.org/wiki/Low-pass_filter#Difference_equation_through_discrete_time_sampling
         */
        data_filtered_ = beta_ * data_filtered_ + (1 - beta_) * data;

        /*
         *  Return filtered data.
         */
        return data_filtered_;
    }

private:

    Type data_filtered_;    //!< Templated filtered data.
    double beta_;    //!< Low-pass filter beta.
};
}   // namespace firmware
}   // namespace biped

#endif  // UTILITY_LOW_PASS_FILTER_H_
