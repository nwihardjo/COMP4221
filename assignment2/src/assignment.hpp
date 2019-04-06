//
// Created by Dekai WU and YAN Yuchen on 20190326.
//

#ifndef FEEDFORWARD_ASSIGNMENT_HPP
#define FEEDFORWARD_ASSIGNMENT_HPP

#include "make_transducer.hpp"

extern const char* STUDENT_ID;
extern const unsigned NUM_EPOCHS;

tg::transducer_t your_classifier(const vector<tg::token_t> &vocab, const vector<tg::postag_t> &postags);

vector<tg::token_t> get_features(const vector<tg::token_t>& sentence, unsigned token_index);

#endif //FEEDFORWARD_ASSIGNMENT_HPP
