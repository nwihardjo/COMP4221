#include "make_transducer.hpp"
#include "util.hpp"
#include "assignment.hpp"
#include "../part_review/util.hpp"
#include "../part_b/util.hpp"
#include <shared_utils.hpp>
#include <chunk_t.hpp>
using namespace tg;
using namespace part_c;

void error_analysis(const vector<vector<symbol_t>> &predicted_postags, const vector<vector<symbol_t>> &predicted_syntac, const vector<vector<symbol_t>> &predicted_iobes_tags, const vector<vector<symbol_t>> &oracles, const vector<vector<symbol_t>> &sentence_tokens, const vector<unsigned> &predicate_pos) {
        int total_count = 0;
        int wrong = 0;

        for (auto i = 0; i < predicted_iobes_tags.size(); i++) {
                auto sent_pred_postags = predicted_postags[i];
		auto sent_pred_syntac = predicted_syntac[i];
                auto sent_pred_iobes_tags = predicted_iobes_tags[i];
                auto sent_oracles = oracles[i];
                auto sent_tokens = sentence_tokens[i];
		auto pred = predicate_pos[i];

                for (auto j = 0; j < sent_pred_iobes_tags.size(); j++) {
                        total_count++;
                        if (sent_pred_iobes_tags[j] != sent_oracles[j]){
                                wrong++;
                                //if (j > 4 && j < sent_pred_iobes_tags.size()-4) {
                                if (j > 0 && int(j) < int(sent_pred_iobes_tags.size())-1){
                                  cout << sent_pred_iobes_tags[j] << "\t" << sent_oracles[j] << "\t";
			          cout << sent_tokens[j] << " (" << sent_pred_postags[j-1] << " " << sent_pred_postags[j]  << " " << sent_pred_postags[j+1] << ", " << sent_pred_syntac[j] << ")\t";
				  cout << "pred: " << sent_tokens[pred] << " (" << sent_pred_postags[pred] << ") " << pred << " " << sent_tokens.size() << endl; 
				}
			       //<< "\t" << sent_pred_postags[j] << endl << "\t";
                               //         for (int i = -4; i < 5; i++) {
                               //                 cout << sent_pred_postags[j+i] << " ";
                               //         }
                               //         cout << endl;
                               //cout << "\t" << sent_tokens[j-1] << " " << sent_pred_postags[j-1] << " " << sent_tokens[j+1] << " " << sent_pred_postags[j+1] << endl;
                       }
		}
	}

        double acc = wrong * 100.0 / total_count;
        cout << "Got wrong: " << wrong << " out of " << total_count << endl;
        cout << "Accuracy of the model " << acc << "%" << endl;
}

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

  cout << "training, please be patient (it will take 10x time when compared with part b)" <<endl;
  classifier.train(training_examples, training_oracles, NUM_EPOCHS);

  auto [test_sents, test_predicate_positions, test_postags] = read_dataset("/project/cl/httpd/htdocs/COMP4221_2019Q1_a3/res/devdata_part_c_iobes.xml");


  cout << "development testing" <<endl;
  vector<vector<symbol_t >> predicted_iobes_tags = srl_sentences_iobes(classifier, postagger, shallow_syntactic_parser, test_sents, test_predicate_positions);

  auto predicted_dev_postags = part_review::postag_sentences(postagger, test_sents);
  auto predicted_syntax_tag = part_b::chunk_sentences_iobes(shallow_syntactic_parser, postagger, test_sents);
  error_analysis(predicted_dev_postags, predicted_syntax_tag, predicted_iobes_tags, test_postags, test_sents, test_predicate_positions);
  report_score(predicted_iobes_tags, test_postags);
  save_dataset("predict_iobes.xml", make_tuple(test_sents, test_predicate_positions, predicted_iobes_tags));
  cout << "prediction saved to predict_iobes.xml"<<endl;

  return 0;
}
