#include "make_transducer.hpp"
#include "util.hpp"
using namespace tg;
using namespace part_review;
int main() {
  auto [train_sents, train_postags] = read_dataset("/project/cl/httpd/htdocs/COMP4221_2019Q1_a3/res/traindata_postag.xml");

  auto [training_examples, training_oracles] = get_examples_and_oracles(train_sents, train_postags);
  // analyze vocabulary and initialize the model
  auto classifier = part_review::your_classifier(collect_vocab_from_symbol_matrix(training_examples), collect_vocab_from_symbol_matrix(train_postags));

  cout << "training data size:"<<training_examples.size()<<endl;

  cout << "training" <<endl;
  classifier.train(convert_to_feature(training_examples), convert_to_feature(training_oracles), part_review::NUM_EPOCHS);

  auto [test_sents, test_postags] = read_dataset("/project/cl/httpd/htdocs/COMP4221_2019Q1_a3/res/devdata_postag.xml");

  cout << "development testing" <<endl;
  auto predicted_postags = postag_sentences(classifier, test_sents);

  cout << "accuracy: " << compute_accuracy(predicted_postags, test_postags) << endl;

  save_dataset("predict.xml", make_pair(test_sents, predicted_postags));
  cout << "prediction saved to predict.xml"<<endl;

  {
    std::ofstream ofs("model.xml");
    if(!ofs.is_open()) throw std::runtime_error("cannot open file for output: model.xml");
    classifier.save(ofs);
    cout << "model saved to model.xml" <<endl;
  }


  return 0;
}
