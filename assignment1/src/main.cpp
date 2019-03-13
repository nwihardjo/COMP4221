#include <iostream>
#include "token_feedforward_classifier.hpp"
#include <fstream>
#include "assignment.hpp"
using namespace std;
int main() {
  ifstream train_ifs("traindata.xml");
  auto [training_tokens, training_oracles] = read_dataset(train_ifs);
  train_ifs.close();

  cout << "initializing" <<endl;
  init(unique(training_tokens), unique(training_oracles));

  cout << "training" <<endl;
  train(training_tokens, training_oracles);

  ifstream dev_ifs("devdata.xml");
  auto [dev_tokens, dev_oracles] = read_dataset(dev_ifs);
  dev_ifs.close();

  cout << "testing on development test data" <<endl;
  auto dev_predicted = test(dev_tokens);

  cout << "development test accuracy:" << compute_accuracy(dev_predicted, dev_oracles) << endl;

  return 0;
}
