#include "make_transducer.hpp"
#include "util.hpp"
#include "assignment.hpp"
#include "../part_review/util.hpp"
#include "../part_b/util.hpp"
#include <shared_utils.hpp>
#include <chunk_t.hpp>
using namespace tg;
using namespace part_c;

int main() {

  auto postag_vocab = collect_vocab_from_symbol_matrix(part_review::read_dataset("/project/cl/httpd/htdocs/COMP4221_2019Q1_a3/res/traindata_postag.xml").second);

  auto syntactic_tags_vocab = collect_vocab_from_symbol_matrix(part_b::read_dataset("/project/cl/httpd/htdocs/COMP4221_2019Q1_a3/res/traindata_part_b_iobes.xml").second);

  auto [train_sents, train_predicate_positions, train_oracles] = part_c::read_dataset("/project/cl/httpd/htdocs/COMP4221_2019Q1_a3/res/traindata_part_c_iobes.xml");

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

  cout << "loading your shallow syntactic parser" << endl;
  transducer_t shallow_syntactic_parser;
  {
    string SHALLOW_SYNTACTIC_PARSER_PATH = "../part_b/model.xml";
    ifstream ifs(SHALLOW_SYNTACTIC_PARSER_PATH);
    if(!ifs.is_open()) throw std::runtime_error("cannot open file for input: " + SHALLOW_SYNTACTIC_PARSER_PATH);
    shallow_syntactic_parser.load(ifs);
  }

  cout << "performing shallow syntactic parse" <<endl;
  vector<vector<symbol_t>> predicted_syntactic_tags = part_b::chunk_sentences_iobes(shallow_syntactic_parser, postagger, train_sents);

  // resolve inconsistency
  for(auto &x:predicted_syntactic_tags) {x = resolve_inconsistency(x);}


  auto [training_examples, training_oracles] = part_c::get_examples_and_oracles(train_sents, train_predicate_positions, predicted_postags, predicted_syntactic_tags, train_oracles);

  // analyze vocabulary and initialize the model
  auto classifier = your_classifier(train_sents, postag_vocab, syntactic_tags_vocab, collect_vocab_from_symbol_matrix(train_oracles));

  cout << "training data size:"<<training_examples.size()<<endl;

  cout << "training, please be patient" <<endl;
  for(unsigned i_epoch = 0; i_epoch < NUM_EPOCHS; ++i_epoch) {
    cout << "epoch #"<< i_epoch <<endl;
    for(unsigned i=0; i<training_examples.size(); i+=1000) {
      cout << "training data #"<<i<<endl;
      if(i + 1000 >= training_examples.size()) {
        classifier.train(
          vector<vector<feature_t>>(training_examples.begin() + i, training_examples.end()),
          vector<feature_t>(training_oracles.begin() + i, training_oracles.end()),
          1);
      }
      else {
        classifier.train(
          vector<vector<feature_t>>(training_examples.begin() + i, training_examples.begin() + i + 1000),
          vector<feature_t>(training_oracles.begin() + i, training_oracles.begin() + i + 1000),
          1);
      }
    }
  }
  auto [test_sents, test_predicate_positions, test_postags] = read_dataset("/project/cl/httpd/htdocs/COMP4221_2019Q1_a3/res/devdata_part_c_iobes.xml");


  cout << "development testing" <<endl;
  vector<vector<symbol_t >> predicted_iobes_tags = srl_sentences_iobes(classifier, postagger, shallow_syntactic_parser, test_sents, test_predicate_positions);

  report_score(predicted_iobes_tags, test_postags);

  save_dataset("predict_iobes.xml", make_tuple(test_sents, test_predicate_positions, predicted_iobes_tags));
  cout << "prediction saved to predict_iobes.xml"<<endl;

  return 0;
}
