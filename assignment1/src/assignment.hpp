//
// Created by Dekai WU and YAN Yuchen on 20190307.
//

#ifndef COMP4221_2019Q1_A1_STUB_ASSIGNMENT_HPP
#define COMP4221_2019Q1_A1_STUB_ASSIGNMENT_HPP
#include "token_feedforward_classifier.hpp"

extern const char* STUDENT_ID;

void init(const std::vector<std::string>& vocab, const std::vector<std::string>& labels);

void train(const std::vector<std::string>& tokens, const std::vector<std::string>& oracles);

std::vector<std::string> test(const std::vector<std::string>& tokens);

#endif
