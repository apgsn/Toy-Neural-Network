#include <ctime>
#include "Net.h"

int main(){
  srand(time(NULL));
  const unsigned FREQ_SHOW_SET = 1000, TRAINING_SETS = 30000;
  const float ALPHA = .3;
  const std::vector<unsigned> TOPOLOGY = {2, 4, 1};
  Net net(TOPOLOGY, ALPHA);

  std::cout << "Neural Network : XOR training\nPress any key to start ";
  std::getchar();
  std::cout <<"\nTraining session\n";
  unsigned set = TRAINING_SETS;
  while(set--){
    bool a = std::rand()%2, b = std::rand()%2;
    std::vector<float> inputs = {(float)a, (float)b};
    net.forwardPropagation(inputs);
    std::vector<float> outputs = {(float)(a ^ b)};
    net.backPropagation(outputs);
    if(!(set%FREQ_SHOW_SET)){
      std::vector<float> results = net.output(2);
      std::cout << "#" << TRAINING_SETS - set << "\tErr. " << std::fixed << std::abs(outputs[0] - results[0]) << std::endl;
    }
  }

  std::cout <<"\nTest session\n";
  for(unsigned i = 0; i < 4; i++){
    bool a = i/2, b = i%2;
    std::vector<float> test = {(float)a, (float)b};
    net.forwardPropagation(test);
    std::vector<float> results = net.output(2);
    std::cout << a << "^" << b << " = " << results[0];
    std::cout << (round(results[0]) == (a ^ b) ? "\tOk" : "\tError") << std::endl;
  }
  return 0;
}
