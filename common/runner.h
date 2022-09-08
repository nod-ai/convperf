#pragma once

namespace convperf {

class Runner {
public:
  Runner() {}
  /*
   * Function for setting up any data structures
   * needed for the computation
   */
  virtual void setup(const float *a, const float *b, float *c) = 0;

  /*
   * Function with the computation that will be
   * profiled
   */
  virtual void run(const float *a, const float *b, float *c) = 0;

  /*
   * Function to store results
   * from computation to output buffer
   *
   */
  virtual void getResults(float *c) = 0;

  virtual ~Runner() = default;
};

}
