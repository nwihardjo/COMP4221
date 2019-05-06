//
// Created by Dekai WU and YAN Yuchen on 20190417.
//

#ifndef COMP4221_2019Q1_A3_CHUNK_T_HPP
#define COMP4221_2019Q1_A3_CHUNK_T_HPP

#include <vector>
#include <variant>
#include <string>
#include <stdexcept>
#include <memory>
#include <set>
#include "default_srl_graph_stub.hpp"

using namespace std;

class chunk_t {
public:
  string name;
  vector<string> tokens;

};

class _chunk_token_generate_iobes_tags {
public:
  vector<string> operator()(const chunk_t &chunk) {
    if (chunk.tokens.empty()) return {};
    string name_as_suffix = chunk.name.empty() ? "" : "-" + chunk.name;
    if (chunk.tokens.size() == 1) return {"S" + name_as_suffix};
    vector<string> ret;
    ret.push_back("B" + name_as_suffix);
    for (unsigned i = 1; i < chunk.tokens.size() - 1; ++i) {
      ret.push_back("I" + name_as_suffix);
    }
    ret.push_back("E" + name_as_suffix);
    return ret;
  }

  vector<string> operator()(const string &token) {
    return {"O"};
  }
};


using chunked_sentence_t = vector<variant<chunk_t, string>>;

inline pair<string, string> break_named_iobes_tag(const string &named_iobes_tag) {
  if (named_iobes_tag.empty()) throw std::runtime_error("break_named_iobes_tag: named iobes tag cannot be empty");
  if (named_iobes_tag.size() == 1) return make_pair(named_iobes_tag, string());
  if (named_iobes_tag[1] != '-')
    throw std::runtime_error("break_named_iobes_tag: malformed named iobes tag: " + named_iobes_tag);
  return make_pair(named_iobes_tag.substr(0, 1), named_iobes_tag.substr(2));
}

inline chunked_sentence_t
parse_iobes_tags(const vector<string> &named_iobes_tags, const vector<string> &sentence = {}) {
  shared_ptr<chunk_t> chunk;
  chunked_sentence_t ret;
  for (long i = 0; i < named_iobes_tags.size(); ++i) {
    const auto &token = (i < sentence.size()) ? sentence[i] : string();
    const auto[iobes_tag, chunk_name] = break_named_iobes_tag(named_iobes_tags[i]);
    if (iobes_tag == "B") {
      if (chunk) {
        ret.push_back(*chunk);
        chunk = nullptr;
      }
      chunk = make_shared<chunk_t>();
      chunk->name = chunk_name;
      chunk->tokens.push_back(token);
    } else if (iobes_tag == "E") {
      if (chunk) {
        if (chunk->name == chunk_name) {
          chunk->tokens.push_back(token);
          ret.push_back(*chunk);
          chunk = nullptr;
        } else {
          ret.push_back(*chunk);
          chunk = nullptr;

          chunk_t tmp;
          tmp.name = chunk_name;
          tmp.tokens.push_back(token);
          ret.push_back(move(tmp));
        }
      } else {
        chunk_t tmp;
        tmp.name = chunk_name;
        tmp.tokens.push_back(token);
        ret.push_back(move(tmp));
      }
    } else if (iobes_tag == "I") {
      if (chunk) {
        if (chunk->name == chunk_name) {
          chunk->tokens.push_back(token);
        } else {
          ret.push_back(*chunk);

          chunk = make_shared<chunk_t>();
          chunk->name = chunk_name;
          chunk->tokens.push_back(token);
        }
      } else {
        chunk = make_shared<chunk_t>();
        chunk->name = chunk_name;
        chunk->tokens.push_back(token);
      }
    } else if (iobes_tag == "S") {
      if (chunk) {
        ret.push_back(*chunk);
        chunk = nullptr;
      }
      chunk_t tmp;
      tmp.name = chunk_name;
      tmp.tokens.push_back(token);
      ret.push_back(move(tmp));
    } else if (iobes_tag == "O") {
      if (chunk) {
        ret.push_back(*chunk);
        chunk = nullptr;
      }
      ret.push_back(token);
    } else {
      throw std::runtime_error("parse_iobes: unknown iobes tag: " + iobes_tag);
    }
  }
  if (chunk) {
    ret.push_back(*chunk);
    chunk = nullptr;
  }
  return ret;
}

inline vector<string> generate_iobes_tags(const chunked_sentence_t &chunked_sentence) {
  auto visitor = _chunk_token_generate_iobes_tags();
  vector<string> ret;
  for (const auto &x:chunked_sentence) {
    vector<string> to_append = visit(visitor, x);
    copy(to_append.begin(), to_append.end(), back_inserter(ret));
  }
  return ret;
}

inline set<tg_stub::labeled_span> get_labeled_spans(const chunked_sentence_t &chunked_sentence) {
  set<tg_stub::labeled_span> ret;
  unsigned i = 0;
  for (const auto &x:chunked_sentence) {
    if (holds_alternative<string>(x)) ++i;
    else {
      auto &chunk = get<chunk_t>(x);
      ret.emplace(chunk.name, i, i + chunk.tokens.size());
      i += chunk.tokens.size();
    }
  }
  return ret;
}

template<typename RANGE_OF_LABELED_SPAN>
vector<string> generate_iobes_tags(unsigned sentence_length, const RANGE_OF_LABELED_SPAN &spans) {
  vector<string> iobes(sentence_length, "O");
  for (const auto &span:spans) {
    for (unsigned i = span.i(); i < span.j(); ++i) {
      if (iobes[i] != "O") throw std::runtime_error("generate_iobes_tags: spans are inconsistent");
    }

    if (span.j() - span.i() <= 1) {
      iobes[span.i()] = "S-" + span.label();
    } else {
      iobes[span.i()] = "B-" + span.label();

      for (unsigned i = span.i() + 1; i < span.j() - 1; ++i) {
        iobes[i] = "I-" + span.label();
      }

      iobes[span.j() - 1] = "E-" + span.label();
    }
  }
  return iobes;
}

vector<string> resolve_inconsistency(const vector<string> &named_iobes_tags) {
  return generate_iobes_tags(parse_iobes_tags(named_iobes_tags));
}

#endif //COMP4221_2019Q1_A3_CHUNK_T_HPP
