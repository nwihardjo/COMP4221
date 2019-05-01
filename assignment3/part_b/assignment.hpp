//
// Created by Dekai WU and YAN Yuchen on 20190417.
//

#ifndef COMP4221_2019Q1_A3_ASSIGNMENT_HPP
#define COMP4221_2019Q1_A3_ASSIGNMENT_HPP

#include <make_transducer.hpp>
using namespace tg;

namespace part_a {
  extern const unsigned NUM_EPOCHS;
  transducer_t your_classifier(const vector<sentence_t> &training_set, const vector<symbol_t> &postags,
                               const vector<symbol_t> &iobes_tags);

  vector<feature_t>
  get_features(const vector<symbol_t> &sentence, const vector<symbol_t> &postags, unsigned target_index);

}

#endif //COMP4221_2019Q1_A3_ASSIGNMENT_HPP
