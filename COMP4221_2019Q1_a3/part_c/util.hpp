//
// Created by Dekai WU and YAN Yuchen on 20190417.
//

#ifndef COMP4221_2019Q1_A3_UTIL_HPP_C
#define COMP4221_2019Q1_A3_UTIL_HPP_C

#include <make_transducer.hpp>
#include <xml_archive.hpp>
#include <fstream>
#include "assignment.hpp"
#include "../part_review/util.hpp"
#include "../part_b/util.hpp"
#include <chunk_t.hpp>
#include "default_srl_graph_stub.hpp"

using namespace tg;
namespace part_c {

  void convert_to_iobes_xml(const string &path_to_srl_xml, const string &path_to_iobes_xml) {
    vector<tg_stub::default_srl_graph> srls;
    {
      ifstream ifs(path_to_srl_xml);
      if (!ifs.is_open()) throw std::runtime_error("cannot read file " + path_to_srl_xml);
      cereal::hltc_xml_input_archive ar(ifs);
      ar >> srls;
    }
    ofstream ofs(path_to_iobes_xml);
    if (!ofs.is_open()) throw std::runtime_error("cannot write file " + path_to_iobes_xml);
    cereal::hltc_xml_output_archive oa(ofs);
    oa.nest("dataset",[&]() {
      for(const auto &srl:srls) {
        const auto &sentence = srl.sen();
        oa.nest("sent", [&]() {
          for(const auto &frame:srl.get_frames()) {
            const auto &pred_span = frame.first;
            const auto &args_span = frame.second;
            oa.attribute("pred_position",pred_span.i());
            oa.nest("frame", [&]() {
              const auto sentence_shallow_semantic_iobes = generate_iobes_tags(sentence.size(), args_span);
              for(unsigned i=0; i<sentence.size(); ++i) {
                oa.attribute("type", sentence_shallow_semantic_iobes[i]);
                oa(cereal::make_nvp("token", sentence[i]));
              }
            });
          }
        });
      }
    });
  }

  tuple<vector<sentence_t>, vector<unsigned>, vector<vector<symbol_t >>> read_dataset(const string &path_to_train_data) {
    tuple<vector<sentence_t>, vector<unsigned>, vector<vector<symbol_t >>> dataset;
    ifstream ifs(path_to_train_data);
    if (!ifs.is_open()) throw std::runtime_error("cannot read file " + path_to_train_data);
    cereal::hltc_xml_input_archive ar(ifs);
    ar.nest([&]() {
      auto &sentences = get<0>(dataset);
      auto &pred_positions = get<1>(dataset);
      auto &tags = get<2>(dataset);
      sentences.clear();
      pred_positions.clear();
      tags.clear();
      while (ar.hasNextChild()) {
        ar.nest("sent", [&]() {
          while(ar.hasNextChild()) {
            unsigned pred_position;
            ar.attribute("pred_position", pred_position);
            pred_positions.push_back(pred_position);
            ar.nest("frame", [&]() {
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
      }
    });
    return dataset;
  }

  pair<vector<vector<feature_t>>, vector<feature_t>>
  get_examples_and_oracles(const vector<sentence_t> &sentences, const vector<unsigned> &pred_positions, const vector<vector<symbol_t>> &postags, const vector<vector<symbol_t>> &syntactic_iobes_tags, const vector<vector<symbol_t>> &semantic_iobes_tags) {
    vector<vector<feature_t>> ret_examples;
    vector<feature_t> ret_oracles;
    for (unsigned i = 0; i < sentences.size(); ++i) {
      auto sentence = sentences[i];
      auto pred_position = pred_positions[i];
      auto sentence_postags = postags[i];
      auto sentence_syntactic_tags = syntactic_iobes_tags[i];
      auto sentence_semantic_tags = semantic_iobes_tags[i];
      for (unsigned j = 0; j < sentence.size(); ++j) {
        ret_examples.push_back(part_c::get_features(sentence, sentence_postags, sentence_syntactic_tags, j, pred_position));
        ret_oracles.emplace_back(sentence_semantic_tags[j]);
      }
    }
    return make_pair(ret_examples, ret_oracles);
  }

  void save_dataset(const string &path, const tuple<vector<sentence_t>, vector<unsigned>, vector<vector<symbol_t >>> &dataset) {
    ofstream ofs(path);
    if (!ofs.is_open()) throw std::runtime_error("cannot write file " + path);
    cereal::hltc_xml_output_archive ar(ofs);
    ar.nest("dataset", [&]() {
      auto &sentences = get<0>(dataset);
      auto &pred_positions = get<1>(dataset);
      auto &tags = get<2>(dataset);
      if (sentences.size() != tags.size())
        throw std::runtime_error("dataset sentences and postags must have equal size");
      for (unsigned i = 0; i < sentences.size(); ++i) {
        ar.attribute("pred_position", pred_positions[i]);
        ar.nest("frame", [&]() {
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

  vector<vector<symbol_t>> srl_sentences_iobes(const transducer_t &shallow_semantic_parser, const transducer_t &postagger, const transducer_t &shallow_syntactic_parser, const vector<sentence_t> &sentences, const vector<unsigned> &pred_positions) {
    auto postags = part_review::postag_sentences(postagger, sentences);
    auto syntactic_tags = part_b::chunk_sentences_iobes(shallow_syntactic_parser, postagger, sentences);

    // resolve inconsistency
    for(auto &x:syntactic_tags) {x = resolve_inconsistency(x);}

    auto predicted_tags_flattened = shallow_semantic_parser.transduce_many(
      get_examples_and_oracles(sentences, pred_positions, postags, syntactic_tags, sentences).first);
    vector<vector<symbol_t >> predicted_tags;
    unsigned flattened_pivot = 0;
    for (const auto &sentence:sentences) {
      vector<symbol_t> sentence_tags;
      for (const auto &token:sentence) {
        sentence_tags.push_back(get<symbol_t>(predicted_tags_flattened[flattened_pivot++][0]));
      }
      predicted_tags.push_back(sentence_tags);
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

    cout << endl;
    cout << "==== token level (before inconsistency resolution) ====" << endl;
    {
      unsigned token_level_correct_count = 0;
      unsigned token_level_total_count = 0;
      for(unsigned i=0; i<predicteds.size(); ++i) {
        const auto &predicted = predicteds[i];
        const auto &oracle = oracles[i];
        if(predicted.size() != oracle.size()) throw std::runtime_error("comptue_f_score: prediction#"+to_string(i)+" and oracle#"+to_string(i)+" must have same size");
        for(unsigned j=0; j<predicted.size(); ++j) {
          if(predicted[j] == oracle[j])  ++token_level_correct_count;
        }
        token_level_total_count += predicted.size();
      }
      cout << "precision = "<< token_level_correct_count/(double)token_level_total_count <<endl;
    }
    cout << endl;

    cout << "==== token level (after inconsistency resolution) ====" << endl;
    {
      unsigned token_level_correct_count = 0;
      unsigned token_level_total_count = 0;
      for(unsigned i=0; i<predicteds.size(); ++i) {
        const auto predicted = generate_iobes_tags(parse_iobes_tags(predicteds[i]));
        const auto oracle = generate_iobes_tags(parse_iobes_tags(oracles[i]));

        if(predicted.size() != oracle.size()) {
          throw std::runtime_error("comptue_f_score: prediction#"+to_string(i)+" and oracle#"+to_string(i)+" must have same size");
        }
        for(unsigned j=0; j<predicted.size(); ++j) {
          if(predicted[j] == oracle[j])  ++token_level_correct_count;
        }
        token_level_total_count += predicted.size();
      }
      cout << "precision = "<< token_level_correct_count/(double)token_level_total_count <<endl;
    }
    cout << endl;


    cout << "==== chunk level ====" << endl;
    unsigned precision_nominator = 0;
    unsigned precision_denominator = 0;
    unsigned recall_nominator = 0;
    unsigned recall_denominator = 0;
    for (unsigned i = 0; i < predicteds.size(); ++i) {
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
    cout << endl;
  }

}
#endif //COMP4221_2019Q1_A3_UTIL_HPP
