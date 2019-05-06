//
// Created by Dekai WU and YAN Yuchen on 20190501.
//

#ifndef SRL_STUB_DEFAULT_SRL_GRAPH_HPP
#define SRL_STUB_DEFAULT_SRL_GRAPH_HPP

#include <string>
#include <vector>
#include <xml_archive.hpp>

namespace tg_stub {
  using namespace std;

  /**
   * represents a sentence segment with a label
   * it contains only the position information of the segment.
   * it does not contain the segment content
   */
  class labeled_span {
    string label_m;
    unsigned i_m{};
    unsigned j_m{};
  public:
    labeled_span() = default;

    labeled_span(const labeled_span &) = default;

    labeled_span(labeled_span &&) noexcept = default;

    labeled_span &operator=(const labeled_span &) = default;

    labeled_span &operator=(labeled_span &&) noexcept = default;

    /**
     * construct a labeled_span with a label, and its position information
     * \param label the label
     * \param i the begin position (inclusive)
     * \param j the end position (inclusive)
     */
    labeled_span(string label, unsigned int i, unsigned int j) : label_m(move(label)), i_m(i), j_m(j) {}

    /**
     * get the begin position
     * \return the begin position (inclusive)
     */
    unsigned i() const { return i_m; }

    /**
     * get the end position
     * \return the end position (exclusive)
     */
    unsigned j() const { return j_m; };

    /**
     * get the label
     * \return the label
     */
    const string &label() const { return label_m; }

    friend bool operator<(const labeled_span &x, const labeled_span &y) {
      return std::tie(x.label_m, x.i_m, x.j_m) < std::tie(y.label_m, y.i_m, y.j_m);
    }
  };

  /**
   * represents an SRL
   */
  class default_srl_graph {
  public:
    using sentence_t = vector<string>;

    /**
     * represents a semantic frame. this pair that contains:
     * * predicate
     * * a list of arguments (role fillers)
     */
    using frame_t = pair<labeled_span, vector<labeled_span>>;
  private:
    sentence_t sen_m;
    vector<frame_t> frames_m;
  public:
    default_srl_graph() = default;

    default_srl_graph(const default_srl_graph &) = default;

    default_srl_graph(default_srl_graph &&) noexcept = default;

    default_srl_graph &operator=(const default_srl_graph &) = default;

    default_srl_graph &operator=(default_srl_graph &&) noexcept = default;

    /**
     * construct a srl graph with sentence and its semantic frames
     * \param sentence the sentence
     * \param frames a list of semantic frames
     */
    default_srl_graph(sentence_t sentence, vector<frame_t> frames) : sen_m(move(sentence)), frames_m(move(frames)) {}

    /**
     * get the sentence
     * \return the sentence, as a list of tokens
     */
    const sentence_t &sen() const {
      return sen_m;
    }

    /**
     * get the list of semantic frames
     * \return a list of semantic frames
     */
    const vector<frame_t> &get_frames() const {
      return frames_m;
    }

    // specialized serialization behavior
    // will be consumed by cereal library, don't call it directly
    void save(cereal::hltc_xml_output_archive &oa) const {
      const auto &sentence = sen();
      for (const auto &frame:get_frames()) {
        const auto &predicate = frame.first;
        const auto &roles = frame.second;
        // put all roles in a map, so that they can be looked up by their starting positions
        unordered_map<unsigned, labeled_span> roles_by_starting_position;
        for (const auto &role:roles) {
          roles_by_starting_position.insert(make_pair(role.i(), role));
        }
        oa.nest("frame", [&]() {

          // for every position in the sentence
          for (unsigned i = 0; i < sentence.size();) {
            if (predicate.i() == i) {
              // if the position contains predicate, serialize the predicate
              oa.attribute("type", predicate.label());
              oa.nest("pred", [&]() {
                for (; i < predicate.j(); ++i) {
                  oa(cereal::make_nvp("token", sentence[i]));
                }
              });
            } else if (roles_by_starting_position.count(i) > 0) {
              // if the position contains role, serialize the role
              const auto &item = roles_by_starting_position.at(i);
              oa.attribute("type", item.label());
              oa.nest("arg", [&]() {
                for (; i < item.j(); ++i) {
                  oa(cereal::make_nvp("token", sentence[i]));
                }
              });
            } else {
              // serialize the token
              oa(cereal::make_nvp("token", sentence[i]));
              ++i;
            }
          }
        });
      }
    }

    // specialized deserialization behavior
    // will be consumed by cereal library, don't call it directly
    void load(cereal::hltc_xml_input_archive &ia) {
      sen_m.clear();
      frames_m.clear();
      bool is_sentence_filled = false;
      while (ia.hasNextChild()) {
        ia.nest("frame", [&]() {
          labeled_span predicate;
          vector<labeled_span> roles;
          for (unsigned i = 0; ia.hasNextChild();) {
            string node_name = ia.getNextChildName();
            if (node_name == "pred") {
              auto type = ia.get_attribute("type");
              auto s = i;
              ia.nest([&]() {
                for (; ia.hasNextChild(); ++i) {
                  string token;
                  ia(token);
                  if (!is_sentence_filled) sen_m.push_back(token);
                }
              });
              predicate = labeled_span(type, s, i);
            } else if (node_name == "arg") {
              auto type = ia.get_attribute("type");
              auto s = i;
              ia.nest([&]() {
                for (; ia.hasNextChild(); ++i) {
                  string token;
                  ia(token);
                  if (!is_sentence_filled) sen_m.push_back(token);
                }
              });
              roles.emplace_back(type, s, i);
            } else if (node_name == "token") {
              string token;
              ia(token);
              if (!is_sentence_filled) sen_m.push_back(token);
              ++i;
            } else {
              throw std::runtime_error("default_srl_graph: error when deserializing: unknown tag name " + node_name);
            }
          }
          is_sentence_filled = true;
          frames_m.emplace_back(predicate, roles);
        });
      }
    }
  };

  // specialized serialization behavior for list of SRLs
  // will be consumed by cereal library, don't call it directly
  inline void save(cereal::hltc_xml_output_archive &oa, const std::vector<default_srl_graph> &srls) {
    for (const auto &srl:srls) {
      oa(cereal::make_nvp("sent", srl));
    }
  }

  // specialized deserialization behavior for list of SRLs
  // will be consumed by cereal library, don't call it directly
  inline void load(cereal::hltc_xml_input_archive &ia, std::vector<default_srl_graph> &srls) {
    while (ia.hasNextChild()) {
      default_srl_graph srl;
      ia(cereal::make_nvp("sent", srl));
      srls.push_back(move(srl));
    }
  }
}

#endif //SRL_STUB_DEFAULT_SRL_GRAPH_HPP
