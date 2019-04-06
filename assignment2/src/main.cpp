#include "make_transducer.hpp"
#include "util.hpp"
using namespace tg;
int main() {
  if(string("00000000") == STUDENT_ID) throw std::runtime_error("please fill in your student ID");

  auto[train_sents, train_postags] = read_dataset("traindata.xml");

  auto [training_examples, training_oracles] = get_examples_and_oracles(train_sents, train_postags);
  // analyze vocabulary and initialize the model
  auto classifier = your_classifier(collect_vocab_from_symbol_matrix(training_examples), collect_vocab_from_symbol_matrix(train_postags));
  cout << "training" <<endl;
  classifier.train(convert_to_feature(training_examples), convert_to_feature(training_oracles), NUM_EPOCHS);

  auto [test_sents, test_postags] = read_dataset("devdata.xml");

  cout << "development testing" <<endl;
  auto predicted_postags_flattened = classifier.transduce_many(convert_to_feature(get_examples_and_oracles(test_sents, test_postags).first));

  vector<vector<postag_t>> predicted_postags;
  unsigned flattened_pivot = 0;
  for (const auto &sentence:test_sents) {
    vector<postag_t> sentence_postags;
    for (const auto &token:sentence) {
      sentence_postags.push_back( get<postag_t>(predicted_postags_flattened[flattened_pivot++][0]));
    }
    predicted_postags.push_back(sentence_postags);
  }

  cout << "accuracy: " << compute_accuracy(predicted_postags, test_postags) << endl;

  save_dataset("predict.xml", make_pair(test_sents, predicted_postags));
  cout << "prediction saved to predict.xml"<<endl;
  return 0;
}
