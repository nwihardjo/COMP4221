//
// Created by Dekai WU and YAN Yuchen on 20190417.
//

#include "assignment.hpp"
#include <shared_utils.hpp>

namespace part_c{
/*
  // =========================================
  // ======== ITERATION 1 - HYPOTHESIS 1 ===== 
  // =========================================
  const unsigned NUM_EPOCHS = 5;

  transducer_t your_classifier(const vector <sentence_t> &training_set, const vector <symbol_t> &postag_vocab, const vector <symbol_t> &iobes_syntactic_tag_vocab, const vector <symbol_t> &iobes_semantic_role_vocab) {
    std::unordered_set<symbol_t> token_set;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        token_set.insert(token);
      }
    }
    vector <symbol_t> vocab(token_set.begin(), token_set.end());

    auto embedding_lookup = make_embedding_lookup(64, vocab);

    auto postag_onehot = make_onehot(postag_vocab);

    auto iobes_syntactic_onehot = make_onehot(iobes_syntactic_tag_vocab);

    auto concatenate = make_concatenate(4);

    auto dense0 = make_dense_feedfwd(64, make_tanh());

    auto dense1 = make_dense_feedfwd(iobes_semantic_role_vocab.size(), make_softmax());

    auto onehot_inverse = make_onehot_inverse(iobes_semantic_role_vocab);

    return compose(
      group(
        embedding_lookup, // handles the input - target token
        embedding_lookup, // handles the input - predicate tokoen
        postag_onehot, // handles the input - target token POS tag
        iobes_syntactic_onehot), // handles the input - target token IOBES shallow syntactic tag
        //make_identity()), // handles the input - predicate-target distance
      concatenate, dense0, 
      dense1, onehot_inverse);
  }

  vector<feature_t> get_features(const vector<token_t> &sentence, const vector<symbol_t> &postags, const vector<symbol_t> &shallow_syntactic_tags, unsigned target_index, unsigned predicate_index) {

    return vector<feature_t>{sentence[target_index], sentence[predicate_index], postags[target_index], shallow_syntactic_tags[target_index]
    };
  }

  // =========================================
  // ======== ITERATION 1 - HYPOTHESIS 2 ===== 
  // =========================================
  
  const unsigned NUM_EPOCHS = 5;
  transducer_t your_classifier(const vector <sentence_t> &training_set, const vector <symbol_t> &postag_vocab, const vector <symbol_t> &iobes_syntactic_tag_vocab, const vector <symbol_t> &iobes_semantic_role_vocab) {
    std::unordered_set<symbol_t> token_set;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        token_set.insert(token);
      }
    }
    vector <symbol_t> vocab(token_set.begin(), token_set.end());

    auto embedding_lookup = make_embedding_lookup(64, vocab);

    auto postag_onehot = make_onehot(postag_vocab);

    auto iobes_syntactic_onehot = make_onehot(iobes_syntactic_tag_vocab);

    auto concatenate = make_concatenate(5);

    auto dense0 = make_dense_feedfwd(64, make_tanh());

    auto dense1 = make_dense_feedfwd(iobes_semantic_role_vocab.size(), make_softmax());

    auto onehot_inverse = make_onehot_inverse(iobes_semantic_role_vocab);

    return compose(
      group(
        embedding_lookup, // handles the input - target token
        embedding_lookup, // handles the input - predicate tokoen
        postag_onehot, // handles the input - target token POS tag
        iobes_syntactic_onehot, // handles the input - target token IOBES shallow syntactic tag
        make_identity()), // handles the input - predicate-target distance
      concatenate, dense0, 
      dense1, onehot_inverse);
  }

  vector<feature_t> get_features(const vector<token_t> &sentence, const vector<symbol_t> &postags, const vector<symbol_t> &shallow_syntactic_tags, unsigned target_index, unsigned predicate_index) {

    return vector<feature_t>{sentence[target_index], sentence[predicate_index], postags[target_index], shallow_syntactic_tags[target_index], (double) target_index - predicate_index};
  }

*/

  // =========================================
  // ======== ITERATION 1 - HYPOTHESIS 3 ===== 
  // =========================================
  //
  const unsigned NUM_EPOCHS = 7;
  transducer_t your_classifier(const vector <sentence_t> &training_set, const vector <symbol_t> &postag_vocab, const vector <symbol_t> &iobes_syntactic_tag_vocab, const vector <symbol_t> &iobes_semantic_role_vocab) {
    std::unordered_set<symbol_t> token_set;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        token_set.insert(token);
      }
    }
    vector <symbol_t> vocab(token_set.begin(), token_set.end());
    auto embedding_lookup = make_embedding_lookup(64, vocab);

    auto postag_onehot = make_onehot(postag_vocab);

    auto iobes_syntactic_onehot = make_onehot(iobes_syntactic_tag_vocab);

    auto concatenate = make_concatenate(10);

    auto dense0 = make_dense_feedfwd(64, make_tanh());
    auto dense_ = make_dense_feedfwd(32, make_tanh());

    auto dense1 = make_dense_feedfwd(iobes_semantic_role_vocab.size(), make_softmax());

    auto onehot_inverse = make_onehot_inverse(iobes_semantic_role_vocab);

    return compose(
      group(
        embedding_lookup, // handles the input - target token
        embedding_lookup, // handles the input - predicate tokoen
	postag_onehot, /// predicate POS tag
	postag_onehot, 
	postag_onehot,
        postag_onehot, // handles the input - target token POS tag
        iobes_syntactic_onehot, // handles the input - target token IOBES shallow syntactic tag
	iobes_syntactic_onehot,
	iobes_syntactic_onehot,
	// make_identity(),
        make_identity()), // handles the input - predicate-target distance
      concatenate, dense0, dense_, 
      dense1, onehot_inverse);
  }

  vector<feature_t> get_features(const vector<token_t> &sentence, const vector<symbol_t> &postags, const vector<symbol_t> &shallow_syntactic_tags, unsigned target_index, unsigned predicate_index) {
	feature_t prev = (target_index > 0) ? postags[target_index-1] : "<s>";
	feature_t next = (int(target_index) < int(sentence.size())-1) ? postags[target_index+1] : "<s>";
	feature_t syn_prev = (target_index > 0) ? shallow_syntactic_tags[target_index-1] : "<s>";
	feature_t syn_next = (int(target_index) < int(shallow_syntactic_tags.size())-1) ? shallow_syntactic_tags[target_index+1] : "<s>";

    return vector<feature_t>{sentence[target_index], sentence[predicate_index], prev, postags[target_index], next, postags[predicate_index], syn_prev, shallow_syntactic_tags[target_index], syn_next, (double) target_index - predicate_index};
  }
}
