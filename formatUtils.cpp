#include "formatUtils.h"

double percentageIncrease(double originalValue, double NewValue) {
   double increase = NewValue - originalValue;
   return increase / originalValue * 100;
}

double percentageDecrease(double originalValue, double NewValue) {
  double decrease = originalValue - NewValue;
  return decrease / originalValue * 100;
}