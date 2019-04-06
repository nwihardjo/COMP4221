//
// Created by Dekai WU and YAN Yuchen on 20190402.
//

#ifndef SRC_MAKE_TRANSDUCER_HPP
#define SRC_MAKE_TRANSDUCER_HPP
#include <functional>
#include <memory>
#include <variant>
using namespace std;
namespace tg {
  using symbol_t = string;
  using token_t = symbol_t;
  using postag_t = symbol_t;
  using feature_t = std::variant<monostate, vector<float>, string>;
  using sentence_t = vector<token_t>;

  class transducer_t {
  public:
    string id_m;

    transducer_t() = default;
    transducer_t(const transducer_t&) = default;
    transducer_t(transducer_t&&) noexcept = default;
    transducer_t &operator=(const transducer_t&) = default;
    transducer_t &operator=(transducer_t&&) noexcept = default;

    explicit transducer_t(const string &id);

    inline vector<feature_t> operator()(const vector<feature_t> &x) const {
      return transduce(x);
    }

    vector<feature_t> transduce(const vector<feature_t> &x) const;

    vector<vector<feature_t>> transduce_many(const vector<vector<feature_t>> &xs) const;

    void
    train(const vector<vector<feature_t>> &examples, const vector<vector<feature_t>> &oracles, unsigned num_epochs);

    void
    train(const vector<vector<feature_t>> &examples, const vector<feature_t> &oracles, unsigned num_epochs);

    void
    train(const vector<feature_t> &examples, const vector<vector<feature_t>> &oracles, unsigned num_epochs);

    void
    train(const vector<feature_t> &examples, const vector<feature_t> &oracles, unsigned num_epochs);
  };

  transducer_t make_identity();

  transducer_t make_copy(unsigned num_copies);

  transducer_t compose(transducer_t f0, transducer_t f1);

  transducer_t group(transducer_t f0, transducer_t f1);

  transducer_t make_symbolic_k_nearest_neighbors_classifier(unsigned k, unsigned num_inputs, const vector<symbol_t> &output_classes);

  transducer_t make_onehot(const vector<symbol_t> &vocab);

  transducer_t make_embedding_lookup(unsigned embedding_size, const vector<symbol_t> &vocab);

  transducer_t make_onehot_inverse(const vector<symbol_t> &vocab);

  transducer_t make_tanh();

  transducer_t make_softmax();

  transducer_t make_concatenate(unsigned num_inputs);

  transducer_t make_dense_feedfwd(unsigned output_dim, transducer_t activation);

  transducer_t make_l2_distance();

  template<typename... Args>
  transducer_t compose(transducer_t f0, transducer_t f1, Args... fs) {
    return compose(f0, compose(f1, fs...));
  }

  template<typename... Args>
  transducer_t group(transducer_t f0, transducer_t f1, Args... fs) {
    return group(f0, group(f1, fs...));
  }
}

#endif //SRC_MAKE_TRANSDUCER_HPP
