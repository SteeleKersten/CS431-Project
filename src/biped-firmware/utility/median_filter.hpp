/**
 *  @file   median_filter.hpp
 *  @author Simon Yu
 *  @date   03/21/2023
 *  @brief  Median filter templated class header.
 *
 *  This file defines and implements the median filter
 *  templated class.
 */

/*
 *  Include guard.
 */
#ifndef UTILITY_MEDIAN_FILTER_H_
#define UTILITY_MEDIAN_FILTER_H_

/*
 *  External headers.
 */
#include <algorithm>
#include <cmath>
#include <list>
#include <vector>

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
 *  @brief  Median filter templated class.
 *
 *  This templated class provides functions for
 *  filtering data using a simple median filter.
 *
 *  Learn more about the median filter here:
 *  https://en.wikipedia.org/wiki/Median_filter#Worked_one-dimensional_example
 */
template<typename Type>
class MedianFilter
{
public:

    /**
     *  @param  window_size Median filter window size.
     *  @brief  Median filter templated class constructor.
     *
     *  This constructor initializes all class member variables and
     *  the median filter window.
     */
    inline
    MedianFilter(const size_t& window_size) : window_size_(window_size)
    {
        /*
         *  Initialize the median filter window.
         */
        for (size_t i = 0; i < window_size; i ++)
        {
            window_.push_front(Type());
        }
    }

    /**
     *  @tparam Type Type of filtered data.
     *  @param  data New data.
     *  @return Templated filtered data.
     *  @brief  Filter data using the median filter.
     *
     *  This function filters the new data using a simple
     *  median filter implementation.
     *
     *  Learn more about the median filter here:
     *  https://en.wikipedia.org/wiki/Median_filter#Worked_one-dimensional_example
     */
    inline Type
    filter(const Type& data)
    {
        /*
         *  Push new data onto the window and pop the oldest data.
         */
        window_.push_front(data);
        window_.pop_back();

        /*
         *  Copy the window into a vector and calculate the median index.
         */
        std::vector<Type> window_sorted(window_.begin(), window_.end());
        const size_t median_index = static_cast<size_t>(std::floor(window_size_ / 2.0));

        /*
         *  Sort the window.
         */
        std::sort(window_sorted.begin(), window_sorted.end());

        /*
         *  Return the median value in the sorted window.
         */
        return window_sorted[median_index];
    }

private:

    std::list<Type> window_;    //!< Median filter window.
    size_t window_size_;    //!< Median filter window size.
};
}   // namespace firmware
}   // namespace biped

#endif  // UTILITY_MEDIAN_FILTER_H_
