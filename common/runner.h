#pragma once

namespace convperf {

class Runner {
public:
  Runner() {}
  /*
   * Function for setting up any data structures
   * needed for the computation
   */
  void setup(const float *a, const float *b, float *c) {}

  /*
   * Function with the computation that will be
   * profiled
   */
  void run(const float *a, const float *b, float *c) {}

  /*
   * Function to store results
   * from computation to output buffer
   *
   */
  void getResults(float *c) {}
};

}
