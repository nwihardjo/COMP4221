//
// Created by Dekai WU and YAN Yuchen on 20190403.
//

#ifndef COMP4221_2019Q1_A2_UTIL_HPP
#define COMP4221_2019Q1_A2_UTIL_HPP

#include "xml_archive.hpp"
#include <fstream>
#include "assignment.hpp"
using namespace tg;

namespace cereal {
  void load(hltc_xml_input_archive &ar, pair<vector<sentence_t>, vector<vector<postag_t>>> &dataset) {
    auto &sentences = dataset.first;
    auto &postags = dataset.second;
    sentences.clear();
    postags.clear();
    while(ar.hasNextChild()) {
      ar.nest("sent", [&]() {
        ar.nest("nonterminal", [&]() {
          sentence_t sentence;
          vector<postag_t> sentence_postags;
          while(ar.hasNextChild()) {
            postag_t postag;
            ar.attribute("value", postag);
            sentence_postags.push_back(postag);
            token_t token;
            ar(token);
            sentence.push_back(token);
          }
          sentences.push_back(sentence);
          postags.push_back(sentence_postags);
        });
      });
    }
  }
  void save(hltc_xml_output_archive &ar, const pair<vector<sentence_t>, vector<vector<postag_t>>> &dataset) {
    const auto &sentences = dataset.first;
    const auto &postags = dataset.second;
    if(sentences.size() != postags.size()) throw std::runtime_error("dataset sentences and postags must have equal size");
    for(unsigned i=0; i<sentences.size(); ++i) {
      ar.nest("sent", [&]() {
        ar.attribute("value", "TOP");
        ar.nest("nonterminal", [&]() {
          const auto& sentence = sentences[i];
          const auto& sentence_postags = postags[i];
          if(sentence.size() != sentence_postags.size()) throw std::runtime_error("dataset sentence token size and its postag size must be equal");
          for(unsigned i=0; i<sentence.size(); ++i) {
            ar.attribute("value", sentence_postags[i]);
            ar(cereal::make_nvp("nonterminal", sentence[i]));
          }
        });
      });
    }
  }
}

// this function will be provided in our starting code
pair<vector<sentence_t>, vector<vector<postag_t>>> read_dataset(const string &path_to_train_data) {
  pair<vector<sentence_t>, vector<vector<postag_t>>> ret;

  {
    ifstream ifs(path_to_train_data);
    if(!ifs.is_open()) throw std::runtime_error("cannot read file "+path_to_train_data);
    cereal::hltc_xml_input_archive ar(ifs);
    ar >> ret;
  }

  return ret;
}

void save_dataset(const string& path, const pair<vector<sentence_t>, vector<vector<postag_t>>> &dataset) {
  ofstream ofs(path);
  if(!ofs.is_open()) throw std::runtime_error("cannot write file "+path);
  cereal::hltc_xml_output_archive ar(ofs);
  ar << dataset;
}

/**
 * a helper function to collect vocabulary from training data
 * collect distinct symbols from a list of symbols that may contain duplicates
 * \param symbols a list of symbols
 * \return the distinct list
 */
vector<symbol_t> collect_vocab(const vector<symbol_t> &symbols) {
  unordered_set<symbol_t> t(symbols.begin(), symbols.end());
  return vector<symbol_t>(t.begin(), t.end());
}

/**
 * a helper function to collect vocabulary from training data
 * collect distinct symbols from a matrix of symbols that may contain duplicates
 * \param symbol_matrix a matrix of symbols
 * \return the distince list
 */
vector<symbol_t> collect_vocab_from_symbol_matrix(const vector<vector<symbol_t>> &symbol_matrix) {
  vector<symbol_t> symbols;
  for (const auto &row:symbol_matrix) {
    copy(row.begin(), row.end(), back_inserter(symbols));
  }
  return collect_vocab(symbols);
}

pair<vector<vector<token_t>>, vector<postag_t>> get_examples_and_oracles(const sentence_t &sentence, const vector<postag_t> &postags) {
  vector<vector<token_t>> ret_examples;
  vector<postag_t> ret_oracles;
  for(unsigned j=0; j<sentence.size(); ++j) {
    ret_examples.push_back(get_features(sentence, j));
    ret_oracles.push_back(postags[j]);
  }
  return make_pair(ret_examples, ret_oracles);
}

pair<vector<vector<token_t>>, vector<postag_t>> get_examples_and_oracles(const vector<sentence_t> &sentences, const vector<vector<postag_t>> &postags) {
  vector<vector<token_t>> ret_examples;
  vector<postag_t> ret_oracles;
  for(unsigned i=0; i<sentences.size(); ++i) {
    auto sentence = sentences[i];
    for(unsigned j=0; j<sentence.size(); ++j) {
      ret_examples.push_back(get_features(sentence, j));
      ret_oracles.push_back(postags[i][j]);
    }
  }
  return make_pair(ret_examples, ret_oracles);
}

feature_t convert_to_feature(const symbol_t& x) {
  return feature_t(x);
}

vector<feature_t> convert_to_feature(const vector<symbol_t>& xs) {
  vector<feature_t> ret;
  ret.reserve(xs.size());
  for(const auto &x:xs) {
    ret.push_back(convert_to_feature(x));
  }
  return ret;
}

vector<vector<feature_t>> convert_to_feature(const vector<vector<symbol_t>>& xs) {
  vector<vector<feature_t>> ret;
  ret.reserve(xs.size());
  for(const auto &x:xs) {
    ret.push_back(convert_to_feature(x));
  }
  return ret;
}

double compute_accuracy(const vector<vector<postag_t>> &predicted_postags, const vector<vector<postag_t>> &oracles) {
  if(predicted_postags.size() != oracles.size()) throw std::runtime_error("compute accuracy: predicted postags size and oracles size should be the same");
  unsigned correct_cnt = 0;
  unsigned total_cnt = 0;
  for(unsigned i=0; i<predicted_postags.size(); ++i) {
    auto sentence_predicted_postags = predicted_postags[i];
    auto sentence_oracles = oracles[i];
    if(sentence_predicted_postags.size() != sentence_oracles.size()) throw std::runtime_error("compute accuracy: sentence predicted postags size and sentence oracles size should be the same");
    for(unsigned i=0; i<sentence_predicted_postags.size(); ++i) {
      if(sentence_predicted_postags[i] == sentence_oracles[i]) ++correct_cnt;
      ++total_cnt;
    }
  }
  return correct_cnt/(double)total_cnt;
}

#endif //COMP4221_2019Q1_A2_UTIL_HPP
