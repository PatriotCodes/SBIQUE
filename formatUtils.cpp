#include "formatUtils.h"

string percentageIncrease(double originalValue, double NewValue) {
    double increase = NewValue - originalValue;
    double increasePerc = increase / originalValue * 100;
    return to_string(increasePerc) + "%";
}

string percentageDecrease(double originalValue, double NewValue) {
  double decrease = originalValue - NewValue;
  double increasePerc = decrease / originalValue * 100;
  return to_string(increasePerc) + "%";
}