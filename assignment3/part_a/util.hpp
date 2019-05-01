//
// Created by Dekai WU and YAN Yuchen on 20190417.
//

#ifndef COMP4221_2019Q1_A3_UTIL_HPP
#define COMP4221_2019Q1_A3_UTIL_HPP

#include <make_transducer.hpp>
#include <xml_archive.hpp>
#include <fstream>
#include "assignment.hpp"
#include "../part_review/util.hpp"
#include <chunk_t.hpp>

using namespace tg;
namespace part_a {

  pair<vector<sentence_t>, vector<vector<symbol_t >>> read_dataset(const string &path_to_train_data) {
    pair<vector<sentence_t>, vector<vector<symbol_t>>> dataset;
    ifstream ifs(path_to_train_data);
    if (!ifs.is_open()) throw std::runtime_error("cannot read file " + path_to_train_data);
    cereal::hltc_xml_input_archive ar(ifs);
    ar.nest([&]() {
      auto &sentences = dataset.first;
      auto &tags = dataset.second;
      sentences.clear();
      tags.clear();
      while (ar.hasNextChild()) {
        ar.nest("sent", [&]() {
          sentence_t sentence;
          vector<symbol_t> sentence_tags;
          while (ar.hasNextChild()) {
            symbol_t postag;
            ar.attribute("type", postag);
            sentence_tags.push_back(postag);
            token_t token;
            ar(token);
            sentence.push_back(token);
          }
          sentences.push_back(sentence);
          tags.push_back(sentence_tags);
        });
      }
    });
    return dataset;
  }

  void save_dataset(const string &path, const pair<vector<sentence_t>, vector<vector<symbol_t >>> &dataset) {
    ofstream ofs(path);
    if (!ofs.is_open()) throw std::runtime_error("cannot write file " + path);
    cereal::hltc_xml_output_archive ar(ofs);
    ar.nest("dataset", [&]() {
      const auto &sentences = dataset.first;
      const auto &tags = dataset.second;
      if (sentences.size() != tags.size())
        throw std::runtime_error("dataset sentences and postags must have equal size");
      for (unsigned i = 0; i < sentences.size(); ++i) {
        ar.nest("sent", [&]() {
          const auto &sentence = sentences[i];
          const auto &sentence_tags = tags[i];
          if (sentence.size() != sentence_tags.size())
            throw std::runtime_error("dataset sentence token size and its postag size must be equal");
          for (unsigned i = 0; i < sentence.size(); ++i) {
            ar.attribute("type", sentence_tags[i]);
            ar(cereal::make_nvp("token", sentence[i]));
          }
        });
      }
    });
  }

  pair<vector<vector<feature_t>>, vector<feature_t>>
  get_examples_and_oracles(const vector<sentence_t> &sentences, const vector<vector<symbol_t>> &postags,
                           const vector<vector<symbol_t>> &iobes_tags) {
    vector<vector<feature_t>> ret_examples;
    vector<feature_t> ret_oracles;
    for (unsigned i = 0; i < sentences.size(); ++i) {
      auto sentence = sentences[i];
      auto sentence_postags = postags[i];
      for (unsigned j = 0; j < sentence.size(); ++j) {
        ret_examples.push_back(part_a::get_features(sentence, sentence_postags, j));
        ret_oracles.emplace_back(iobes_tags[i][j]);
      }
    }
    return make_pair(ret_examples, ret_oracles);
  }

  vector<vector<symbol_t >> chunk_sentences_iobes(const transducer_t &chunker, const transducer_t &postagger,
                                                  const vector<sentence_t> &sentences) {
    auto postags = part_review::postag_sentences(postagger, sentences);
    auto predicted_tags_flattened = chunker.transduce_many(
      get_examples_and_oracles(sentences, postags, sentences).first);

    vector<vector<symbol_t >> predicted_tags;
    unsigned flattened_pivot = 0;
    for (const auto &sentence:sentences) {
      vector<symbol_t> sentence_postags;
      for (const auto &token:sentence) {
        sentence_postags.push_back(get<symbol_t>(predicted_tags_flattened[flattened_pivot++][0]));
      }
      predicted_tags.push_back(sentence_postags);
    }

    return predicted_tags;
  }

  pair<unsigned, unsigned>
  count_precision_recall_nominator(const vector<symbol_t> &predicted, const vector<symbol_t> &oracle) {
    auto predicted_labeled_spans = get_labeled_spans(parse_iobes_tags(predicted));
    auto oracle_labeled_spans = get_labeled_spans(parse_iobes_tags(oracle));
    unsigned precision_counts = 0;
    for (const auto &span:predicted_labeled_spans) {
      if (oracle_labeled_spans.count(span) > 0) precision_counts++;
    }
    unsigned recall_counts = 0;
    for (const auto &span:oracle_labeled_spans) {
      if (predicted_labeled_spans.count(span) > 0) recall_counts++;
    }
    return make_pair(precision_counts, recall_counts);
  }

  void report_score(vector<vector<symbol_t>> &predicteds, const vector<vector<symbol_t>> &oracles) {
    if (predicteds.size() != oracles.size())
      throw std::runtime_error("compute_f_score: prediction and oracles must have same size");
    unsigned precision_nominator = 0;
    unsigned precision_denominator = 0;
    unsigned recall_nominator = 0;
    unsigned recall_denominator = 0;
    for (long i = 0; i < predicteds.size(); ++i) {
      const auto &predicted = predicteds[i];
      const auto &oracle = oracles[i];
      auto predicted_labeled_spans = get_labeled_spans(parse_iobes_tags(predicted));
      auto oracle_labeled_spans = get_labeled_spans(parse_iobes_tags(oracle));

      for (const auto &span:predicted_labeled_spans) {
        if (oracle_labeled_spans.count(span) > 0) precision_nominator++;
      }
      precision_denominator += predicted_labeled_spans.size();

      for (const auto &span:oracle_labeled_spans) {
        if (predicted_labeled_spans.count(span) > 0) recall_nominator++;
      }
      recall_denominator += oracle_labeled_spans.size();
    }

    double precision = precision_denominator == 0 ? 1 : precision_nominator / (double) precision_denominator;
    double recall = recall_denominator == 0 ? 1 : recall_nominator / (double) recall_denominator;
    double f = 2*precision*recall/(precision + recall);
    cout << "precision = "<<precision<<"   recall = "<<recall << "   f-score = "<<f<<endl;
  }

}
#endif //COMP4221_2019Q1_A3_UTIL_HPP
