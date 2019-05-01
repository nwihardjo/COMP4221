#include "make_transducer.hpp"
#include "util.hpp"
#include "assignment.hpp"
#include "../part_review/util.hpp"
#include <shared_utils.hpp>
#include <chunk_t.hpp>
using namespace tg;
using namespace part_a;

int main() {

  auto postag_vocab = collect_vocab_from_symbol_matrix(part_review::read_dataset("/project/cl/httpd/htdocs/COMP4221_2019Q1_a3/res/traindata_postag.xml").second);

  auto [train_sents, train_iobes_tags] = read_dataset("/project/cl/httpd/htdocs/COMP4221_2019Q1_a3/res/traindata_part_a_iobes.xml");

  cout << "loading your POS tagger"<<endl;
  transducer_t postagger;
  {
    string POSTAG_MODEL_PATH = "../part_review/model.xml";
    ifstream ifs(POSTAG_MODEL_PATH);
    if(!ifs.is_open()) throw std::runtime_error("cannot open file for input: " + POSTAG_MODEL_PATH);
    postagger.load(ifs);
  }

  cout << "predicting POS tags with your POS tagger" <<endl;
  vector<vector<symbol_t>> predicted_postags = part_review::postag_sentences(postagger, train_sents);

  auto [training_examples, training_oracles] = get_examples_and_oracles(train_sents, predicted_postags, train_iobes_tags);

  // analyze vocabulary and initialize the model
  auto classifier = your_classifier(train_sents, postag_vocab, collect_vocab_from_symbol_matrix(train_iobes_tags));

  cout << "training data size:"<<training_examples.size()<<endl;

  cout << "training" <<endl;
  classifier.train(training_examples, training_oracles, NUM_EPOCHS);

  auto [test_sents, test_postags] = read_dataset("/project/cl/httpd/htdocs/COMP4221_2019Q1_a3/res/devdata_part_a_iobes.xml");


  cout << "development testing" <<endl;
  vector<vector<symbol_t >> predicted_iobes_tags = chunk_sentences_iobes(classifier, postagger, test_sents);

  report_score(predicted_iobes_tags, test_postags);

  save_dataset("predict_iobes.xml", make_pair(test_sents, predicted_iobes_tags));
  cout << "prediction saved to predict_iobes.xml"<<endl;

  // for part A, no need to save the model

//  {
//    std::ofstream ofs("model.xml");
//    if(!ofs.is_open()) throw std::runtime_error("cannot open file for output: model.xml");
//    classifier.save(ofs);
//    cout << "model saved to model.xml" <<endl;
//  }

  return 0;
}
