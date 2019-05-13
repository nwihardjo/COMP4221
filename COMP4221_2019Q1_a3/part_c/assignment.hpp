//
// Created by Dekai WU and YAN Yuchen on 20190417.
//

#ifndef COMP4221_2019Q1_A3_ASSIGNMENT_HPP
#define COMP4221_2019Q1_A3_ASSIGNMENT_HPP

#include <make_transducer.hpp>
using namespace tg;

namespace part_c {
  extern const unsigned NUM_EPOCHS;
  transducer_t your_classifier(const vector <sentence_t> &training_set, const vector <symbol_t> &postag_vocab,
                                            const vector <symbol_t> &iobes_syntactic_tag_vocab,
                                            const vector <symbol_t> &iobes_semantic_role_vocab);

  vector<feature_t> get_features(const vector<token_t> &sentence, const vector<symbol_t> &postags,
                                 const vector<symbol_t> &shallow_syntactic_tags, unsigned target_index,
                                 unsigned predicate_index);

}

#endif //COMP4221_2019Q1_A3_ASSIGNMENT_HPP
